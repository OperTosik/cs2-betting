# cs2-betting

`cs2-betting` is two-agent framework, based on supervised and reinforcement alghoritms.

### Agents

Agent 1: CatBoost classifier is used for predict probably of win team_A. The probability calibration prevents the agent 2 from overbetting.

Agent 2: Proximal Policy Optimization algorithm return actions:

* action 0: not bet
* action 1: bet 1% of bankroll
* action 2: bet 2% of bankroll
* action 3: bet 5% of bankroll

### Hybrid ML architecture (SL + RL)
```
cs2_betting/
├─ Dockerfile
├─ docker-compose.yml
├─ requirements.txt
├─ config/
│  ├─ config.py
├─ src/
│  ├─ prepareData.py
│  ├─ agentSL.py     # CatBoost model
│  ├─ agentSLCalibrated.py
│  ├─ agentRL.py     # PPO agent
│  ├─ env.py         # Gym env
│  ├─ trainSL.py
│  ├─ trainRL.py
│  ├─ predict.py     # interface
│  └─ utils.py
├─ datasets/
│  ├─ BlastBounty2026Season1.csv
│  ├─ IEMKrakow2026.csv
│  ├─ commonData.csv
│  └─ predictData.csv
└─ models/
   ├─ catboost_model.cbm
   └─ pro_model.zip

```

### Installation
<summary>Clone this repository</summary>

```
git clone https://github.com/OperTosik/cs2-betting.git
```
### Data

I have attached the files as an example of dataset. [Dataset description](https://github.com/OperTosik/cs2-betting/blob/main/datasets/README.md)

In `datasets/predictData.csv` you shoud to enter the details of the upcoming matches. I also recommend entering mirrored string for each match like this:
```
2026-01-28,Aurora,GamerLegion,Mirage,3,1,1.38,2.8,100,143,0.533,0.579,2.772588722239781,2.833213344056216,0.5,0.5,0.5714285714285714,0.0019999999999997797,0.27049999999999974,0.54,0.71,0.479,1,1,
2026-01-28,GamerLegion,Aurora,Mirage,3,1,2.8,1.38,-100,-143,0.579,0.533,2.833213344056216,2.772588722239781,0.5,0.5,-0.5714285714285714,-0.0019999999999997797,-0.27049999999999974,0.71,0.54,0.521,1,1,
```

### Configuration

The parameters used are listed in `config/config.py`. List `DATA` is contaned name of dataframe files. You can change features of dataframe in `FEATURES`.

### Create build

```
Docker build -t cs2-betting .
```

### Prepare data

Merge all files from `config.DATA` in one.

```
docker-compose run prepare_data
```

### Training

Train CatBoost model:

```
docker-compose run train_supervised
```

Calibration model:

```
docker-compose run train_sl_cal
```

Train PRO:

```
docker-compose run train_rl
```

### Run app

```
docker-compose run predict
```
