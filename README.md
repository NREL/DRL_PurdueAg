## DRL agent setup

Presetup: Install anaconda https://conda.io/projects/conda/en/latest/user-guide/install/windows.html 

1. Clone the github repository
```
git clone git@github.com:NREL/DRL_PurdueAg.git
```
2. Create virtual environment

   ```
   conda create --name DRL_PurdueAg
   conda activate DRL_PurdueAg
   conda config --env --add channels conda-forge
   conda config --env --set channel_priority strict
   conda env update --name DRL_PurdueAg --file environment.yml
   ```
3. Run rollout script
   ```
   python agent_rollout.py
   ```

4. Create ssh key and put public key on github.com 

