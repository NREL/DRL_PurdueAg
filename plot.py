import numpy as np
import matplotlib.pyplot as plt

class Plot:
    def __init__(self,*args,**kwargs) -> None:
        for dict in args:
            for k, v in dict.items():
                setattr(self,k,v)
        for k, v in kwargs.items():
            setattr(self,k,v)
            
    def plot_action(self, plot_modes: bool = False, variables=None):
        if variables == None:
            fig, ax = plt.subplots(nrows=1,ncols=1,figsize = (10,3))
            color1='tab:red'
            color2='tab:olive'
            ax.plot(self.pHP, color=color1, label='pHP')
            ax.plot(self.pMP, color=color2, label='pMP')
            ax.legend()
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Pressure [bar]')
            if plot_modes:
                color3 = 'black'
                ax2 = ax.twinx()
                ax2.plot(self.Bulk_mode, ls = 'solid',
                color=color3, label = 'Bulk_mode')
                ax2.plot(self.Alt_mode, ls = 'dotted',
                color=color3, label = 'Alt_mode')
                ax2.plot(self.Vac_mode, ls = 'dashed',
                color=color3, label = 'Vac_mode')
                ax2.plot(self.Fert_mode, ls = 'dashdot',
                color=color3, label = 'Fert_mode')
                ax2.set_ylabel('Actuator mode')
                ax2.set_yticks([1,2,3])
                ax2.legend()
            plt.tight_layout()
            plt.show()
        if variables is not None:
            colors=['tab:red', 'tab:olive']
            style = ['solid','dotted','dashed','dashdot']
            fig, ax = plt.subplots(nrows=1,ncols=1,figsize = (10,3))
            for index, variable in enumerate(variables):
                if 'mode' not in variable:             
                    if variable == 'pHP':
                        ax.plot(self.pHP, color=colors[index], 
                        label = variable)
                    elif variable == 'pMP':
                        ax.plot(self.pMP, color=colors[index], 
                        label = variable)
                    ax.legend()
                    ax.set_xlabel('Time [s]')
                    ax.set_ylabel('Pressure [bar]')
                else:
                    ax2 = ax.twinx()
                    color3 = 'black'
                    if variable == 'Bulk_mode':
                        ax2.plot(self.Bulk_mode, 
                        ls = style[index],color=color3, label = variable )
                    elif variable =='Alt_mode':
                        ax2.plot(self.Alt_mode, 
                        ls = style[index],color=color3, label = variable)
                    elif variable =='Vac_mode':
                        ax2.plot(self.Vac_mode, 
                        ls = style[index],color=color3, label = variable)
                    elif variable =='Fert_mode':
                        ax2.plot(self.Fert_mode, 
                        ls = style[index],color=color3, label = variable)
                    ax2.legend()
                    ax2.set_ylabel('Actuator mode')
                    ax2.set_yticks([1,2,3])
            plt.tight_layout()
            plt.show()
    
    def plot_state(self, variables = None, 
                   pressures = False, flowrates = False, 
                   RPM = False):
        if variables == None:
            fig, ax = plt.subplots(nrows=1,ncols=1,figsize = (12,3))
            colors = ['tab:red','tab:orange','tab:olive','tab:green']
            if pressures:
                ax.plot(self.Bulk_P, color=colors[0], label = 'Bulk_P')
                ax.plot(self.Alt_P, color=colors[1], label = 'Alternator_P')
                ax.plot(self.Vac_P, color=colors[2], label = 'Vacuum_P')
                ax.plot(self.Fert_P, color=colors[3], label = 'Fertilizer_P')
                ax.set_xlabel('Time [s]')
                ax.set_ylabel('Pressure [bar]')
                ax.legend()
            
            elif flowrates:
                ax.plot(self.Bulk_Q, color=colors[0], label = 'Bulk_Q')
                ax.plot(self.Alt_Q, color=colors[1], label = 'Alternator_Q')
                ax.plot(self.Vac_Q, color=colors[2], label = 'Vacuum_Q')
                ax.plot(self.Fert_Q, color=colors[3], label = 'Fertilizer_Q')
                ax.set_xlabel('Time [s]')
                ax.set_ylabel('Flow rate [Lpm]')
                ax.legend()

            elif RPM:
                ax.plot(self.Bulk_rpm_delta, color=colors[0], label='Bulk_rpm')
                ax.plot(self.Alt_rpm_delta, color=colors[1], label='Alt_rpm')
                ax.plot(self.Vac_rpm_delta, color=colors[2], label='Vac_rpm')
                ax.plot(self.Fert_rpm_delta, color=colors[3], label='Fert_rpm')
                ax.set_xlabel('Time [s]')
                ax.set_ylabel('Flow rate [Lpm]')
                ax.legend()
            else:
                raise ValueError("Input list of variables")
            
        if variables is not None:
            colors=['tab:red', 'tab:olive', 'maroon','skyblue',
                    'tab:green','navyblue','black','darkgoldenrod']
            style = ['solid','dotted','dashed','dashdot']
            fig, ax = plt.subplots(nrows=1,ncols=1,figsize = (12,3))
            for index, variable in enumerate(variables):  
                if '_P' in variable:         
                    if variable == 'Bulk_P':
                        ax.plot(self.Bulk_P, color=colors[index],label=variable)
                    elif variable == 'Alt_P':
                        ax.plot(self.Alt_P, color=colors[index],label=variable)
                    elif variable == 'Vac_P':
                        ax.plot(self.Vac_P, color=colors[index],label=variable)
                    elif variable == 'Fert_P':
                        ax.plot(self.Fert_P, color=colors[index],label=variable)
                    ax.legend()
                    ax.set_xlabel('Time [s]')
                    ax.set_ylabel('Pressure [bar]')
                elif '_Q' in variable:
                    ax2 = ax.twinx()
                    if variable == 'Bulk_Q':
                        ax2.plot(self.Bulk_Q, color=colors[index], label = variable)
                    elif variable == 'Alt_Q':
                        ax2.plot(self.Alt_P, color=colors[index], label = variable)
                    elif variable == 'Vac_Q':
                        ax2.plot(self.Vac_Q, color=colors[index], label = variable)
                    elif variable == 'Fert_Q':
                        ax2.plot(self.Fert_Q, color=colors[index], label = variable)
                    ax2.legend()
                    ax2.set_ylabel('Actuator flow rate [LPM]')

                elif '_rpm' in variable:
                    ax2 = ax.twinx()
                    if variable == 'Bulk_rpm_delta':
                        ax2.plot(self.Bulk_rpm_delta, color=colors[index], label = variable)
                    elif variable == 'Alt_rpm_delta':
                        ax2.plot(self.Alt_rpm_delta, color=colors[index], label = variable)
                    elif variable == 'Vac_rpm_delta':
                        ax2.plot(self.Vac_rpm_delta, color=colors[index], label = variable)
                    elif variable == 'Fert_rpm_delta':
                        ax2.plot(self.Fert_rpm_delta, color=colors[index], label = variable)
                    ax2.legend()
                    ax2.set_ylabel('Actuator RPM')

            plt.tight_layout()
            plt.show()

    def plot_cmdRPM(self, variables=None):
        if variables == None:
            fig, ax = plt.subplots(nrows=1,ncols=1,figsize = (12,3))
            colors=['tab:red', 'tab:olive', 'maroon','skyblue',
                    'tab:green','navyblue','black','darkgoldenrod']
            style = ['solid','dotted','dashed','dashdot']
            ax.plot(self.Bulk_rpm_cmd, color=colors[0], label = 'Bulk_cmdRPM')
            ax.plot(self.Alt_rpm_cmd, color=colors[1], label = 'Alt_cmdRPM')
            ax.plot(self.Vac_rpm_cmd, color=colors[2], label = 'Vac_cmdRPM')
            ax.plot(self.Fert_rpm_cmd, color=colors[3], label = 'Fert_cmdRPM')
            ax.legend()
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Commanded RPM')

        if variables is not None:  
            fig, ax = plt.subplots(nrows=1,ncols=1,figsize = (12,3))        
            for index, variable in enumerate(variables):
                if variable == 'Bulk_rpm_cmd':
                    ax.plot(self.Bulk_rpm_cmd, color=colors[index],
                    label=variable)
                elif variable == 'Alt_rpm_cmd':
                    ax.plot(self.Alt_rpm_cmd, color=colors[index],
                    label=variable)
                elif variable == 'Vac_rpm_cmd':
                    ax.plot(self.Vac_rpm_cmd, color=colors[index],
                    label=variable)
                elif variable == 'Fert_rpm_cmd':
                    ax.plot(self.Fert_rpm_cmd, color=colors[index],
                    label=variable)

            ax.legend()
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Commanded RPM')
            plt.tight_layout()
            plt.show()