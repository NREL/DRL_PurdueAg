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
              
        obs_low: float = -1e6*np.ones((12,1))
        self.num_actuators: int = 4
        self.num_actuatormodes: int = 3
        self.max_delta_rpm: float = 4000               #[rpm]
        self.max_operating_flowrate: float = 50        #[lpm]
        self.max_operating_pressure: float = 300       #[bar]
        self.min_rail_pressure: float = 20             #[bar]
        self.max_rpm: npt.NDArray[np.float64] = 2*np.array([4998,4950,3915.66,3000]) # max rpm for each actuator (bulk, vac, alt, fert)
        d_x: int = 22
        d_f: int = 12

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
        self.sf = utils.load_scale_factors('/Users/sakhtar/projects/DRL_PurdueAg/model/scale_factors.h5')
        # Load time and amesim simulation data
        self.T, self.data = utils.load_data('/Users/sakhtar/projects/DRL_PurdueAg/data/data.h5')

  
        
        # Load ROM 
        load_model_path = '/Users/sakhtar/projects/DRL_PurdueAg/model/rom.h5'
        self.headers_in = ['pHP', 'pMP', 'Bulk_mode', 'Alt_mode', 'Vac_mode', 'Fert_mode']
        self.headers_env = ['Bulk_rpm_cmd', 'Alt_rpm_cmd', 'Vac_rpm_cmd', 'Fert_rpm_cmd']
        self.headers_out = ['Bulk_P', 'Alt_P', 'Vac_P', 'Fert_P', 
                    'Bulk_Q', 'Alt_Q', 'Vac_Q', 'Fert_Q',
                    'Bulk_rpm_delta', 'Alt_rpm_delta', 'Vac_rpm_delta', 'Fert_rpm_delta']

        headers = {'in':self.headers_in, 'env': self.headers_env, 'out': self.headers_out}
        headers_norm = self.headers_in + self.headers_env + self.headers_out
        #TODO: Normalize self.data here
        self.data, self.sf = utils.norm_data(self.data, 
                                    headers = headers_norm, 
                                    scale_type='minmax')

        self.ROM = rom.ROM(d_x, d_f, 
                        scale_factors=self.sf, 
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
            'pHP': spaces.Box(low=0, 
                              high=1, 
                              shape=(1,), 
                              dtype=np.float64),
            # Action space to calculate medium pressure rail
            'pressure_ratio':spaces.Box(low=0, 
                                          high = 1, 
                                          shape=(1,), 
                                          dtype = np.float64)
        })

        ## Define the observation space: bounds, space type and shape

        # Defining upper limit on observations
        obs_high = np.array([[self.max_operating_pressure] for _ in range(self.num_actuators)] + 
                            [[self.max_operating_flowrate] for _ in range(self.num_actuators)] +
                            [[self.max_delta_rpm] for _ in range(self.num_actuators)])
        
        #Defining observation space
        self.observation_space = spaces.Box(low=np.zeros((12,1)), 
                                            high=np.ones((12,1)), 
                                            dtype=np.float32)

        ## Define exogenous variable

        # Check for timestep in data
        timesteparray_data = np.round(np.diff(self.T),2)
        trutharray, = np.where(timesteparray_data != timesteparray_data[0])
        if trutharray.size == 0:
            self.dt_data = timesteparray_data[0]
        else:
            raise ValueError('Time step is not uniform in the imported data')
        # Compute commanded rpm timeseries
        self.cmd_keys = ['Bulk_rpm_cmd', 'Alt_rpm_cmd',
                    'Vac_rpm_cmd', 'Fert_rpm_cmd']
        self.cmd_rpm_dict = {k: self.data[k] for k in self.cmd_keys}
        self.time_array = self.T
        self.commanded_RPM = np.array([self.cmd_rpm_dict[self.cmd_keys[0]][0,:],
                                       self.cmd_rpm_dict[self.cmd_keys[1]][0,:],
                                       self.cmd_rpm_dict[self.cmd_keys[2]][0,:],
                                       self.cmd_rpm_dict[self.cmd_keys[3]][0,:]])
        

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
        ## TODO: Normalize here as well
        # Sample high and medium pressure levels at time t = 0
        self.low_pressure = self.min_rail_pressure/self.max_operating_pressure
        self.high_pressure = np.random.uniform(low=self.low_pressure, 
                                               high=self.max_operating_pressure)/self.max_operating_pressure
        self.medium_pressure = np.random.uniform(low = self.low_pressure, 
                                                 high = self.high_pressure)*self.high_pressure
        

        # Sample actuator modes randomly at time t =0 
        self.actuator_modes = np.random.randint(low=1,high=4)*np.ones(self.num_actuators)

        
        # Return the random initial observation from ROM model

        # action = Dict('discrete': NumPy (4, t), 'continuous': NumPy (2, t))
        #                order: Bulk_mode, Alt_mode, Vac_mode, Fert_mode
        #                order: pHP, pMP
        action = {'discrete': self.actuator_modes.reshape(4,1),
                'continuous': np.array([[self.high_pressure],[self.medium_pressure]])}
        cmd_rpm = {'continuous': np.array([self.commanded_RPM[0,0],
                                           self.commanded_RPM[1,0],
                                           self.commanded_RPM[2,0],
                                           self.commanded_RPM[3,0]]).reshape(4,1)}

        # obs = Dict('continuous': NumPy (12,))
        #                order: Bulk_P, Alt_P, Vac_P, Fert_P, 
        #                       Bulk_Q, Alt_Q, Vac_Q, Fert_Q,
        #                       Bulk_rpm_delta, Alt_rpm_delta, Vac_rpm_delta, Fert_rpm_delta
        # Compute random obs from ROM
        obs = self.ROM.call_ROM(act=action, cmd=cmd_rpm, 
                                obs=self.init_obs, T=self.dt, 
                                normalized=True)
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
        
        ## exogenous variable
        cmd_rpm = {'continuous': np.array([np.interp(self.simulation_time, fp = self.commanded_RPM[0,:], xp = self.time_array),
                                           np.interp(self.simulation_time, fp = self.commanded_RPM[1,:], xp = self.time_array),
                                           np.interp(self.simulation_time, fp = self.commanded_RPM[2,:], xp = self.time_array),
                                           np.interp(self.simulation_time, fp = self.commanded_RPM[3,:], xp = self.time_array)]).reshape(4,1)}
        
        ## Compute next observation from action and exogenous variables
        next_obs = self.ROM.call_ROM(act=action_input, cmd=cmd_rpm, 
                                     obs=current_obs, T=self.dt,
                                     normalized=True)

        ## Check if the excess pressure is positive or not
        ## if positive for all actuators, mode in the action is valid.
        ## if negative for any actuator, choose a different mode for that actuator, that would results in a higher pressure drop


        ## Compute the reward
        ### Define excess pressure first 
        self.low_pressure = self.min_rail_pressure/self.max_operating_pressure
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
        rpm_deviation = next_obs['continuous'][8:12]
        reward = -( np.sum(excess_pressure*abs(flow_rates))/self.num_actuators \
                    + abs(np.sum(rpm_deviation)) )
                
        
        ## Compute done
        # Simulation time in seconds
        self.simulation_time +=self.dt
        done=False
        if self.simulation_time >= self.episode_simulation_time:
            done = True
        
        # Return the observation, reward, done flag, and additional information

        self.current_obs = next_obs['continuous']

        ## Unnormlize observation, action, and exogenous variables
        # Unnormalize observations
        current_obs_dict = {}
        for index, obs_label in enumerate(self.headers_out):
            current_obs_dict[obs_label] = self.current_obs[index,:]

        current_obs_unnormed = utils.unnorm_data(current_obs_dict,
                                                 headers=self.headers_out,
                                                 scale_factor=self.sf)
        
        # Unnormalize action
        action_continuous = action_input['continuous']
        action_dict = {}
        for index, act_label in enumerate(['pHP', 'pMP']):
            action_dict[act_label] = action_continuous[index]

        action_unnormed_continuous = utils.unnorm_data(action_dict,
                                            headers=['pHP', 'pMP'],
                                            scale_factor=self.sf)
        
        # action_unnormed = np.concatenate((action_unnormed_continuous,
        #                                   action_input['discrete']), axis = 0)
        
        #Unnormalize cmd rpm
        commanded_RPM_dict = {}
        for index, cmd in enumerate(self.cmd_keys):
            commanded_RPM_dict[cmd] = cmd_rpm['continuous'][index,:]

        cmd_rpm_unnormed = utils.unnorm_data(commanded_RPM_dict, 
                                             headers=self.headers_env, 
                                             scale_factor = self.sf)

        info = {'Commanded RPM' : cmd_rpm_unnormed,
                'Action_normed' : action_continuous,
                'Action_continuous' : action_unnormed_continuous,
                'Action_discrete' : action_input['continuous'],
                'Observation' : current_obs_unnormed,
                'Excess pressure': excess_pressure}  # Additional information (if any)

        return self.current_obs, reward, done, info

    def render(self, mode='human'):
        # Define how to render the environment (optional)
        pass