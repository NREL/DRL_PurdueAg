import os
import tensorflow as tf
from pathlib import Path
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
import gym
from gym import spaces
from ray import tune
import rom
import utils

class OptimControllerEnv(gym.Env):
    def __init__(self, env_config=None):
        if env_config is None:
            env_config = dict()

        ## Defining bounds on observations  
              
        obs_low = -1e6*np.ones((12,1))
        self.num_actuators = 4
        self.num_actuatormodes = 3
        self.max_delta_rpm = 4000               #[rpm]
        self.max_operating_flowrate = 50        #[lpm]
        self.max_operating_pressure = 300       #[bar]
        self.min_rail_pressure = 20             #[bar]
        self.max_rpm = 2*np.array([4998,4950,3915.66,3000]) # max rpm for each actuator (bulk, vac, alt, fert)
        d_x = 22
        d_f = 12

        ## Setup episode simulation variables

        self.current_obs = None
        # Total simulation time of the episode[s]
        self.episode_simulation_time = 300 
        # Actual simulation time of the episode [s]
        self.simulation_time = 0
        # Timestep [s]
        self.dt=1
        # Number of timesteps
        self.num_timesteps: int = int(self.episode_simulation_time/self.dt)

        ## Initializing ROM 

        # Import scale_factors disctionary from rom
        sf = utils.load_scale_factors('/Users/sakhtar/projects/DRL_PurdueAg/model/scale_factors.h5')
        # Load time and amesim simulation data
        self.T, self.data = utils.load_data('/Users/sakhtar/projects/DRL_PurdueAg/data/data.h5')

        #TODO: Normalize self.data here
        # Load ROM 
        load_model_path = '/Users/sakhtar/projects/DRL_PurdueAg/model/rom.h5'
        headers_in = ['pHP', 'pMP', 'Bulk_mode', 'Alt_mode', 'Vac_mode', 'Fert_mode']
        headers_env = ['Bulk_rpm_cmd', 'Alt_rpm_cmd', 'Vac_rpm_cmd', 'Fert_rpm_cmd']
        headers_out = ['Bulk_P', 'Alt_P', 'Vac_P', 'Fert_P', 
                    'Bulk_Q', 'Alt_Q', 'Vac_Q', 'Fert_Q',
                    'Bulk_rpm_delta', 'Alt_rpm_delta', 'Vac_rpm_delta', 'Fert_rpm_delta']

        headers = {'in':headers_in, 'env': headers_env, 'out': headers_out}

        self.ROM = rom.ROM(d_x, d_f, 
                        scale_factors=sf, 
                        model_path=load_model_path,
                        headers=headers)
        
        ## Define the action space: Bounds, space type, shape

        self.action_space = spaces.Dict({
            # Discrete action with 4 options corresponding to 3 modes of the actuators
            'actuator_mode': spaces.MultiDiscrete([self.num_actuatormodes, 
                                                   self.num_actuatormodes, 
                                                   self.num_actuatormodes, 
                                                   self.num_actuatormodes]),  
            # Action space for high pressure rail
            'pHP': spaces.Box(low=self.min_rail_pressure, 
                              high=self.max_operating_pressure, 
                              shape=(1,)),
            # Action space fto calculate medium pressure rail
            'pressure_ratio':spaces.Box(low=0, 
                                          high = 1, 
                                          shape=(1,))
        })

        ## Define the observation space: bounds, space type and shape

        # Defining upper limit on observations
        obs_high = np.array([[self.max_operating_pressure] for _ in range(self.num_actuators)] + 
                            [[self.max_operating_flowrate] for _ in range(self.num_actuators)] +
                            [[self.max_delta_rpm] for _ in range(self.num_actuators)])
        
        #Defining observation space
        self.observation_space = spaces.Box(low=-np.inf*np.ones((12,1)), 
                                            high=np.inf*np.ones((12,1)),)

        ## Define exogenous variable
        ## TODO: Change commanded rpm here to match input data (done)


        # Check for timestep in data
        timesteparray_data = np.round(np.diff(self.T),2)
        trutharray, = np.where(timesteparray_data != timesteparray_data[0])
        if trutharray.size == 0:
            self.dt_data = timesteparray_data[0]
        else:
            raise ValueError('Time step is not uniform in the imported data')
        # Compute commanded rpm timeseries
        cmd_keys = ['Bulk_rpm_cmd', 'Alt_rpm_cmd',
                    'Vac_rpm_cmd', 'Fert_rpm_cmd']
        self.cmd_rpm_dict = {k: self.data[k] for k in cmd_keys}
        self.time_array = self.T
        self.commanded_RPM = np.array([self.cmd_rpm_dict[cmd_keys[0]][0,:],
                                       self.cmd_rpm_dict[cmd_keys[1]][0,:],
                                       self.cmd_rpm_dict[cmd_keys[2]][0,:],
                                       self.cmd_rpm_dict[cmd_keys[3]][0,:]])

    def reset(self):
        # Reset the environment state to the initial state
        #Initial observation
        Bulk_P = self.data['Bulk_P']
        Alt_P = self.data['Alt_P']
        Vac_P = self.data['Vac_P']
        Fert_P = self.data['Fert_P']
        Bulk_Q = self.data['Bulk_Q']
        Alt_Q = self.data['Alt_Q']
        Vac_Q = self.data['Vac_Q']
        Fert_Q = self.data['Fert_Q']
        Bulk_rpm_delta = self.data['Bulk_rpm_delta']
        Alt_rpm_delta = self.data['Alt_rpm_delta']
        Vac_rpm_delta = self.data['Vac_rpm_delta']
        Fert_rpm_delta = self.data['Fert_rpm_delta']
        k=np.random.randint(0,Bulk_P.shape[0])
        p=np.random.randint(0,Bulk_P.shape[1])
        self.init_obs = {'continuous': np.array([Bulk_P[k, p], Alt_P[k, p], Vac_P[k, p], Fert_P[k, p],
                                     Bulk_Q[k, p], Alt_Q[k, p], Vac_Q[k, p], Fert_Q[k, p],
                                     Bulk_rpm_delta[k, p], Alt_rpm_delta[k, p], 
                                     Vac_rpm_delta[k, p], Fert_rpm_delta[k, p]]).reshape(12,1)}
        # dt = 1 # time in seconds

        # Random action

        # Sample high and medium pressure levels at time t = 0
        self.low_pressure = self.min_rail_pressure
        self.high_pressure = np.random.uniform(low=self.low_pressure, high=self.max_operating_pressure)
        self.medium_pressure = max(self.low_pressure, np.random.uniform()*self.high_pressure)
        

        # Sample actuator modes randomly at time t =0 
        self.actuator_modes = np.random.randint(low=1,high=4)*np.ones(self.num_actuators)

        
        # Return the random initial observation from ROM model

        # action = Dict('discrete': NumPy (4, t), 'continuous': NumPy (2, t))
        #                order: Bulk_mode, Alt_mode, Vac_mode, Fert_mode
        #                order: pHP, pMP
        action = {'discrete': self.actuator_modes.reshape(4,1),
                'continuous': np.array([[self.high_pressure],[self.medium_pressure]])}

        # cmd_rpm = Dict('continuous': NumPy (4, t))
        #                order: Bulk_rpm, Alt_rpm, Vac_rpm, Fert_rpm

        # Sample commanded RPM randomly at time t = 0. The commanded RPM is determined by the crops etc.
        #TODO: Put actual CMD RPM data for each actuator (done)
        # self.commanded_RPM = np.random.uniform(low=0, high=self.max_delta_rpm)*np.ones(self.num_actuators)
        #Initial cmd rpm
        cmd_rpm = {'continuous': np.array([self.commanded_RPM[0,0],
                                           self.commanded_RPM[1,0],
                                           self.commanded_RPM[2,0],
                                           self.commanded_RPM[3,0]]).reshape(4,1)}

        # obs = Dict('continuous': NumPy (12,))
        #                order: Bulk_P, Alt_P, Vac_P, Fert_P, 
        #                       Bulk_Q, Alt_Q, Vac_Q, Fert_Q,
        #                       Bulk_rpm_delta, Alt_rpm_delta, Vac_rpm_delta, Fert_rpm_delta
        # NOTE: Q values are set to 0 for now

        # Compute random obs from ROM
        obs = self.ROM.call_ROM(act=action, cmd=cmd_rpm, obs=self.init_obs, T=self.dt)
        self.current_obs = obs['continuous']

        return self.current_obs

    def step(self, action):
        """
        Returns: Given current observation and action, returns the next observation, the reward, done, and additional information
        """
        ## Convert action to the form that ROM can take the input in 

        pHP = action['pHP']
        pMP = pHP*action['pressure_ratio']
        actuator_modes = action['actuator_mode']+1
        action_input = {'discrete': actuator_modes.reshape(4,1),
                'continuous': np.array([pHP,pMP])}
        current_obs = {'continuous':self.current_obs}
        
        ## environment variable

        # cmd_rpm = {'continuous': self.commanded_RPM.reshape(4,1)}
       
        cmd_rpm = {'continuous': np.array([np.interp(self.simulation_time, fp = self.commanded_RPM[0,:], xp = self.time_array),
                                           np.interp(self.simulation_time, fp = self.commanded_RPM[1,:], xp = self.time_array),
                                           np.interp(self.simulation_time, fp = self.commanded_RPM[2,:], xp = self.time_array),
                                           np.interp(self.simulation_time, fp = self.commanded_RPM[3,:], xp = self.time_array)]).reshape(4,1)}
        
        ## Compute next observation from action and exogenous variables
        next_obs = self.ROM.call_ROM(act=action_input, cmd=cmd_rpm, obs=current_obs, T=self.dt)

        ## Check if the excess pressure is positive or not
        ## if positive for all actuators, mode in the action is valid.
        ## if negative for any actuator, choose a different mode for that actuator, that would results in a higher pressure drop


        ## Compute the reward
        ### Define excess pressure first 
        pLP = self.low_pressure
        available_pressure = np.array([])
        for i in actuator_modes:
            if i ==1:
                available_pressure_val = pMP-pLP
                available_pressure = np.append(available_pressure, available_pressure_val)
            elif i==2:
                available_pressure_val = pHP-pMP
                available_pressure = np.append(available_pressure, available_pressure_val)
            elif i==3:
                available_pressure_val = pHP-pLP
                available_pressure = np.append(available_pressure, available_pressure_val)
            else:
                raise ValueError("Invalid value for mode. Should be an integer in the list [1,2,3]")

        excess_pressure = available_pressure.reshape(4,1)-next_obs['continuous'][0:4].reshape(4,1)
        flow_rates = next_obs['continuous'][4:8].reshape(4,1)
        rpm_deviation = np.sum(next_obs['continuous'][8:12])
        reward = -(np.sum(excess_pressure*abs(flow_rates)) \
                    + abs(rpm_deviation))
                
        
        ## Compute done
        # Simulation time in seconds
        self.simulation_time +=self.dt
        done=False
        if self.simulation_time >= self.episode_simulation_time:
            done = True
        
        # Return the observation, reward, done flag, and additional information

        self.current_obs = next_obs['continuous']
        info = {'Commanded RPM' : cmd_rpm['continuous'],
                'Action' : action_input,
                'Observation' : self.current_obs,
                'Excess pressure': excess_pressure}  # Additional information (if any)

        return self.current_obs, reward, done, info

    def render(self, mode='human'):
        # Define how to render the environment (optional)
        pass