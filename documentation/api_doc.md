Great! Here's how I can start building the API documentation based on the information you've provided:

### **API Overview**
   - **Purpose:** The Intelligent Liquidity Provisioning (ILP) Framework V2 is designed to optimize liquidity provisioning strategies for Uniswap V3 pools using reinforcement learning (RL) techniques.
   - **Key Features:**
     - PPO and DDPG agents for RL.
     - Uniswap V3 smart Contract Interface
     - Agent Based Model to Train Agents
     - Django application for API interaction


### **Endpoints and Operations**

   #### General Endpoints
   - **`GET /api/v1/`**  
     - **Description:** Returns a basic welcome message indicating the framework is running.
     - **Example Response:**
       ```json
       "Intelligent Liquidity Provisioning Framework"
       ```
   - **`POST /api/v1/initialize-script/`**  
     - **Description:** Initializes the script by setting the base path and resetting environment variables.
     - **Parameters:** 
       - `base_path`: The base path for the framework.
       - `reset_env_var`: Boolean to reset environment variables.
     - **Example Request:**
       ```json
       {
         "base_path": "<path_to_your_cloned_repository>",
         "reset_env_var": true
       }
       ```
     - **Example Response:**
       ```json
       {
         "message": "Base path and reset env var configured successfully"
       }
       ```
    - **`GET /api/v1/file/download/`**
        - **Description:** Downloads a specified file from the server.
        - **Parameters:**
        - `file_path`: The relative path to the directory where the file is stored (e.g., `model_outdir_csv/ddpg/ppo_prod1`).
        - `file_name`: The name of the file to download (e.g., `train_logs.csv`).
        - **Example Request:**
        ```bash
        curl -X GET "https://ilp.tempestfinance.xyz/api/v1/file/download/?file_path=model_outdir_csv/ddpg/ppo_prod1&file_name=train_logs.csv" -O
        ```
        - **Example Response:**
        - The file is downloaded to the user's machine.
        - **Error Handling:**
        - If the file does not exist, the server responds with a `400 Bad Request` error.


   #### DDPG Endpoints
   - **`POST /api/v1/ddpg/train/`**
     - **Description:** Trains the DDPG agent.
     - **Parameters:** Include details like `max_steps`, `n_episodes`, and others related to training.
     - **Example Response:**
       ```json
       {
         "status": "running",
         "ddpg_actor_model_path": "",
         "ddpg_critic_model_path": "",
         "max_steps": 0,
         "time_consumed": 0
       }
       ```

   - **`GET /api/v1/ddpg/train-result/`**
     - **Description:** Retrieves the result of the DDPG training process.
     - **Example Response:**
       ```json
       {
         "status": "done",
         "ddpg_actor_model_path": "path/to/model",
         "ddpg_critic_model_path": "path/to/model",
         "max_steps": 1000,
         "time_consumed": 3600
       }
       ```

   - **`POST /api/v1/ddpg/evaluate/`**
     - **Description:** Evaluates the DDPG agent.
     - **Parameters:** Evaluation parameters like `eval_steps`, `eval_episodes`, etc.
     - **Example Response:**
       ```json
       {
         "message": "Evaluation performed"
       }
       ```

   #### PPO Endpoints
   - **`POST /api/v1/ppo/train/`**
     - **Description:** Trains the PPO agent.
     - **Parameters:** Include details like `max_steps`, `n_episodes`, and others related to training.
     - **Example Response:**
       ```json
       {
         "ppo_actor_model_path": "path/to/model",
         "ppo_critic_model_path": "path/to/model"
       }
       ```

   - **`POST /api/v1/ppo/evaluate/`**
     - **Description:** Evaluates the PPO agent.
     - **Parameters:** Evaluation parameters like `eval_steps`, `eval_episodes`, etc.
     - **Example Response:**
       ```json
       {
         "message": "Evaluation performed"
       }
       ```

   #### Inference Endpoints
   - **`POST /api/v1/inference/`**
     - **Description:** Performs inference using trained models based on the Uniswap V3 model and user-defined parameters.
     - **Parameters:** Pool state, user preferences, pool ID, etc.
     - **Example Response:**
       ```json
       {
         "strategy_action": "action",
         "ddpg_action": "action",
         "ppo_action": "action"
       }
       ```

   - **`POST /api/v1/predict-action/`**
     - **Description:** Predicts the action using the DDPG and PPO models.
     - **Parameters:** Similar to the inference endpoint.
     - **Example Response:**
       ```json
       {
         "ddpg_action": "action",
         "ppo_action": "action"
       }
       ```

### **Data Models**
   - **Training Request Model:** Parameters like `max_steps`, `n_episodes`, etc.
   - **Evaluation Request Model:** Parameters like `eval_steps`, `eval_episodes`, etc.
   - **Inference Request Model:** Parameters like `pool_state`, `user_preferences`, etc.

### Extracting Training and Evaluation Data

After training and evaluating the models using the Intelligent Liquidity Provisioning (ILP) Framework, users can extract and analyze the training and evaluation logs that are stored as CSV files. These logs provide detailed insights into the training and evaluation processes, enabling users to perform further analysis, visualize performance metrics, and refine their strategies.

#### **General Structure of Log Files**

The ILP Framework stores the training and evaluation logs in a specified directory within the project. The logs are saved in CSV format, making them easy to manipulate and analyze using tools like Python (pandas), Excel, or any other data analysis tool.

Each model type (DDPG and PPO) has its own sub-directory where the logs are saved. The structure is as follows:

```
/<base_directory>/model_outdir_csv/ddpg/<model_name>/train_logs.csv
/<base_directory>/model_outdir_csv/ddpg/<model_name>/eval_logs.csv

/<base_directory>/model_outdir_csv/ppo/<model_name>/train_logs.csv
/<base_directory>/model_outdir_csv/ppo/<model_name>/eval_logs.csv
```

Here, `<base_directory>` is the root directory of your ILP Framework project, and `<model_name>` is the name you assigned to your model during training (e.g., `ppo_prod1`).

#### **Example Paths for Log Files**

- **DDPG Model Logs**
  - Training Logs: `<base_directory>/model_outdir_csv/ddpg/<model_name>/train_logs.csv`
  - Evaluation Logs: `<base_directory>/model_outdir_csv/ddpg/<model_name>/eval_logs.csv`

- **PPO Model Logs**
  - Training Logs: `<base_directory>/model_outdir_csv/ppo/<model_name>/train_logs.csv`
  - Evaluation Logs: `<base_directory>/model_outdir_csv/ppo/<model_name>/eval_logs.csv`


**Download the Files via API :**
   If your ILP Framework instance is running on a remote server, you can download the CSV logs using the API. The `download_file` endpoint allows you to download files from the server to your local machine.

   Example curl command to download a training log:

   ```bash
   curl -X GET "https://ilp.tempestfinance.xyz/api/v1/file/download/?file_path=model_outdir_csv/ddpg/ppo_prod1&file_name=train_logs.csv" -O
   ```

   This command will save the file as `train_logs.csv` in your current directory.

#### **Common Use Cases**

- **Performance Tracking:** Use the training logs to track the model's performance over time, such as loss, accuracy, or any custom metrics.
- **Hyperparameter Tuning:** Evaluate how changes in hyperparameters affect the model's performance using the evaluation logs.
- **Strategy Optimization:** Use the insights gained from the logs to refine and optimize your liquidity provisioning strategies.

By understanding how to access and analyze these logs, users can gain deeper insights into the training and evaluation processes, ultimately improving the effectiveness of their models.

### Example Usage

#### **1. Initialize Script**

To set the base path and reset environment variables for the framework, use the following command:

```bash
curl -X POST https://ilp.tempestfinance.xyz/api/v1/initialize-script/ \
-H "Content-Type: application/json" \
-d '{
    "base_path": "<path_to_your_cloned_repository>",
    "reset_env_var": true
}'
```

Replace `<path_to_your_cloned_repository>` with the actual path where the repository is located on your system.

#### **2. Train DDPG Agent**

To train the DDPG agent with custom parameters:

```bash
curl -X POST https://ilp.tempestfinance.xyz/api/v1/ddpg/train/ \
-H "Content-Type: application/json" \
-d '{
    "max_steps": 2,
    "n_episodes": 2,
    "model_name": "model_storage/ddpg/ddpg_fazool",
    "alpha": 0.001,
    "beta": 0.001,
    "tau": 0.8,
    "batch_size": 50,
    "training": true,
    "agent_budget_usd": 10000,
    "use_running_statistics": false
}'
```

#### **3. Evaluate DDPG Agent**

To evaluate the DDPG agent:

```bash
curl -X POST https://ilp.tempestfinance.xyz/api/v1/ddpg/evaluate/ \
-H "Content-Type: application/json" \
-d '{
    "eval_steps": 2,
    "eval_episodes": 2,
    "model_name": "model_storage/ddpg/ddpg_fazool",
    "percentage_range": 0.6,
    "agent_budget_usd": 10000,
    "use_running_statistics": false
}'
```

#### **4. Train PPO Agent**

To train the PPO agent with custom parameters:

```bash
curl -X POST https://ilp.tempestfinance.xyz/api/v1/ppo/train/ \
-H "Content-Type: application/json" \
-d '{
    "max_steps": 2,
    "n_episodes": 2,
    "model_name": "model_storage/ppo/ppo2_fazool22",
    "buffer_size": 5,
    "n_epochs": 20,
    "gamma": 0.5,
    "alpha": 0.001,
    "gae_lambda": 0.75,
    "policy_clip": 0.6,
    "max_grad_norm": 0.6,
    "agent_budget_usd": 10000,
    "use_running_statistics": false,
    "action_transform": "linear"
}'
```

#### **5. Evaluate PPO Agent**

To evaluate the PPO agent:

```bash
curl -X POST https://ilp.tempestfinance.xyz/api/v1/ppo/evaluate/ \
-H "Content-Type: application/json" \
-d '{
    "eval_steps": 2,
    "eval_episodes": 2,
    "model_name": "model_storage/ppo/ppo2_fazool",
    "percentage_range": 0.5,
    "agent_budget_usd": 10000,
    "use_running_statistics": false,
    "action_transform": "linear"
}'
```

#### **6. Perform Inference**

To perform inference using the model, providing details about the pool state and user preferences:

```bash
curl -X POST https://ilp.tempestfinance.xyz/api/v1/inference/ \
-H "Content-Type: application/json" \
-d '{
    "pool_state": {
        "current_profit": 500,
        "price_out_of_range": false,
        "time_since_last_adjustment": 40000,
        "pool_volatility": 0.2
    },
    "user_preferences": {
        "risk_tolerance": {"profit_taking": 50, "stop_loss": -500},
        "investment_horizon": 7,
        "liquidity_preference": {"adjust_on_price_out_of_range": true},
        "risk_aversion_threshold": 0.1,
        "user_status": "new_user"
    },
    "pool_id": "0x4e68ccd3e89f51c3074ca5072bbac773960dfa36",
    "ddpg_agent_path": "model_storage/ddpg/ddpg_1",
    "ppo_agent_path": "model_storage/ppo/lstm_actor_critic_batch_norm"
}'
```

#### **7. Predict Action**

To predict the action using the DDPG and PPO models:

```bash
curl -X POST https://ilp.tempestfinance.xyz/api/v1/predict-action/ \
-H "Content-Type: application/json" \
-d '{
    "pool_id": "0x4e68ccd3e89f51c3074ca5072bbac773960dfa36",
    "ddpg_agent_path": "model_storage/ddpg/ddpg_1",
    "ppo_agent_path": "model_storage/ppo/lstm_actor_critic_batch_norm",
    "date_str": "2024-05-05"
}'
```
#### **8. Download Log Files**

To download the training or evaluation log files:

```bash
# Download training logs for a DDPG model
curl -X GET "https://ilp.tempestfinance.xyz/api/v1/file/download/?file_path=model_outdir_csv/ddpg/ppo_prod1&file_name=train_logs.csv" -O

# Download evaluation logs for a PPO model
curl -X GET "https://ilp.tempestfinance.xyz/api/v1/file/download/?file_path=model_outdir_csv/ppo/ppo_prod1&file_name=eval_logs.csv" -O
```



## **Parameters Explanation**

### **1. Initialize Script (`POST /api/v1/initialize-script/`)**

- **`base_path`**
  - **Description:** The absolute or relative path to the base directory where the ILP Framework is set up.
  - **Data Type:** `string`
  - **Required:** Yes
  - **Example:** `"/path/to/your/cloned/repository"`

- **`reset_env_var`**
  - **Description:** A boolean flag indicating whether to reset environment variables.
  - **Data Type:** `boolean`
  - **Required:** Yes
  - **Example:** `true`

### **2. Train DDPG Agent (`POST /api/v1/ddpg/train/`)**

- **`max_steps`**
  - **Description:** The maximum number of steps to train the DDPG agent.
  - **Data Type:** `integer`
  - **Required:** Yes
  - **Default Value:** `10000`
  - **Example:** `5000`

- **`n_episodes`**
  - **Description:** The number of episodes to run during training.
  - **Data Type:** `integer`
  - **Required:** Yes
  - **Default Value:** `100`
  - **Example:** `50`

- **`model_name`**
  - **Description:** The name under which the trained model will be saved.
  - **Data Type:** `string`
  - **Required:** Yes
  - **Example:** `"model_storage/ddpg/ddpg_model"`

- **`alpha`**
  - **Description:** The learning rate for the DDPG agent's actor network.
  - **Data Type:** `float`
  - **Required:** Yes
  - **Default Value:** `0.001`
  - **Example:** `0.0005`

- **`beta`**
  - **Description:** The learning rate for the DDPG agent's critic network.
  - **Data Type:** `float`
  - **Required:** Yes
  - **Default Value:** `0.001`
  - **Example:** `0.0005`

- **`tau`**
  - **Description:** The soft update parameter for updating target networks.
  - **Data Type:** `float`
  - **Required:** Yes
  - **Default Value:** `0.005`
  - **Example:** `0.01`

- **`batch_size`**
  - **Description:** The size of the batch used during training.
  - **Data Type:** `integer`
  - **Required:** Yes
  - **Default Value:** `64`
  - **Example:** `128`

- **`training`**
  - **Description:** A boolean flag indicating whether to run the model in training mode.
  - **Data Type:** `boolean`
  - **Required:** Yes
  - **Default Value:** `true`
  - **Example:** `true`

- **`agent_budget_usd`**
  - **Description:** The budget in USD allocated to the agent for liquidity provisioning.
  - **Data Type:** `float`
  - **Required:** Yes
  - **Default Value:** `10000.0`
  - **Example:** `5000.0`

- **`use_running_statistics`**
  - **Description:** A boolean flag indicating whether to use running statistics during training.
  - **Data Type:** `boolean`
  - **Required:** No
  - **Default Value:** `false`
  - **Example:** `true`

### **3. Evaluate DDPG Agent (`POST /api/v1/ddpg/evaluate/`)**

- **`eval_steps`**
  - **Description:** The number of steps to perform during evaluation.
  - **Data Type:** `integer`
  - **Required:** Yes
  - **Default Value:** `1000`
  - **Example:** `500`

- **`eval_episodes`**
  - **Description:** The number of episodes to run during evaluation.
  - **Data Type:** `integer`
  - **Required:** Yes
  - **Default Value:** `100`
  - **Example:** `50`

- **`model_name`**
  - **Description:** The name of the model to be evaluated.
  - **Data Type:** `string`
  - **Required:** Yes
  - **Example:** `"model_storage/ddpg/ddpg_model"`

- **`percentage_range`**
  - **Description:** The percentage range used for evaluating the agent's performance.
  - **Data Type:** `float`
  - **Required:** Yes
  - **Default Value:** `0.5`
  - **Example:** `0.7`

- **`agent_budget_usd`**
  - **Description:** The budget in USD allocated to the agent during evaluation.
  - **Data Type:** `float`
  - **Required:** Yes
  - **Default Value:** `10000.0`
  - **Example:** `5000.0`

- **`use_running_statistics`**
  - **Description:** A boolean flag indicating whether to use running statistics during evaluation.
  - **Data Type:** `boolean`
  - **Required:** No
  - **Default Value:** `false`
  - **Example:** `true`

### **4. Train PPO Agent (`POST /api/v1/ppo/train/`)**

- **`max_steps`**
  - **Description:** The maximum number of steps to train the PPO agent.
  - **Data Type:** `integer`
  - **Required:** Yes
  - **Default Value:** `10000`
  - **Example:** `5000`

- **`n_episodes`**
  - **Description:** The number of episodes to run during training.
  - **Data Type:** `integer`
  - **Required:** Yes
  - **Default Value:** `100`
  - **Example:** `50`

- **`model_name`**
  - **Description:** The name under which the trained model will be saved.
  - **Data Type:** `string`
  - **Required:** Yes
  - **Example:** `"model_storage/ppo/ppo_model"`

- **`buffer_size`**
  - **Description:** The size of the buffer for storing experience.
  - **Data Type:** `integer`
  - **Required:** Yes
  - **Default Value:** `2048`
  - **Example:** `4096`

- **`n_epochs`**
  - **Description:** The number of epochs to run during each update.
  - **Data Type:** `integer`
  - **Required:** Yes
  - **Default Value:** `10`
  - **Example:** `20`

- **`gamma`**
  - **Description:** The discount factor for future rewards.
  - **Data Type:** `float`
  - **Required:** Yes
  - **Default Value:** `0.99`
  - **Example:** `0.95`

- **`alpha`**
  - **Description:** The learning rate for the PPO agent's actor network.
  - **Data Type:** `float`
  - **Required:** Yes
  - **Default Value:** `0.0003`
  - **Example:** `0.0001`

- **`gae_lambda`**
  - **Description:** The lambda parameter for Generalized Advantage Estimation.
  - **Data Type:** `float`
  - **Required:** Yes
  - **Default Value:** `0.95`
  - **Example:** `0.9`

- **`policy_clip`**
  - **Description:** The clipping parameter for the PPO objective function.
  - **Data Type:** `float`
  - **Required:** Yes
  - **Default Value:** `0.2`
  - **Example:** `0.1`

- **`max_grad_norm`**
  - **Description:** The maximum value for gradient clipping.
  - **Data Type:** `float`
  - **Required:** Yes
  - **Default Value:** `0.5`
  - **Example:** `0.3`

- **`agent_budget_usd`**
  - **Description:** The budget in USD allocated to the agent for liquidity provisioning.
  - **Data Type:** `float`
  - **Required:** Yes
  - **Default Value:** `10000.0`
  - **Example:** `5000.0`

- **`use_running_statistics`**
  - **Description:** A boolean flag indicating whether to use running statistics during training.
  - **Data Type:** `boolean`
  - **Required:** No
  - **Default Value:** `false`
  - **Example:** `true`

- **`action_transform`**
  - **Description:** The type of transformation applied to actions, typically `linear` or `non-linear`.
  - **Data Type:** `string`
  - **Required:** Yes
  - **Default Value:** `"linear"`
  - **Example:** `"non-linear"`

### **5. Evaluate PPO Agent (`POST /api/v1/ppo/evaluate/`)**

- **`eval_steps`**
  - **Description:** The number of steps to perform during evaluation.
  - **Data Type:** `integer`
  - **Required:** Yes
  - **Default Value:** `1000`
  - **Example:** `500`

- **`eval_episodes`**
  - **Description:** The number of episodes to run during evaluation.
  - **Data Type:** `integer`
  - **Required:** Yes
  - **Default Value:** `100`
  - **Example:** `50`



- **`model_name`**
  - **Description:** The name of the model to be evaluated.
  - **Data Type:** `string`
  - **Required:** Yes
  - **Example:** `"model_storage/ppo/ppo_model"`

- **`percentage_range`**
  - **Description:** The percentage range used for evaluating the agent's performance.
  - **Data Type:** `float`
  - **Required:** Yes
  - **Default Value:** `0.5`
  - **Example:** `0.7`

- **`agent_budget_usd`**
  - **Description:** The budget in USD allocated to the agent during evaluation.
  - **Data Type:** `float`
  - **Required:** Yes
  - **Default Value:** `10000.0`
  - **Example:** `5000.0`

- **`use_running_statistics`**
  - **Description:** A boolean flag indicating whether to use running statistics during evaluation.
  - **Data Type:** `boolean`
  - **Required:** No
  - **Default Value:** `false`
  - **Example:** `true`

- **`action_transform`**
  - **Description:** The type of transformation applied to actions, typically `linear` or `non-linear`.
  - **Data Type:** `string`
  - **Required:** Yes
  - **Default Value:** `"linear"`
  - **Example:** `"non-linear"`

### **6. Perform Inference (`POST /api/v1/inference/`)**

- **`pool_state`**
  - **Description:** The current state of the liquidity pool, including metrics like current profit, volatility, and time since last adjustment.
  - **Data Type:** `object`
  - **Required:** Yes
  - **Example:**
    ```json
    {
      "current_profit": 500,
      "price_out_of_range": false,
      "time_since_last_adjustment": 40000,
      "pool_volatility": 0.2
    }
    ```

- **`user_preferences`**
  - **Description:** User-defined preferences including risk tolerance, investment horizon, and liquidity preferences.
  - **Data Type:** `object`
  - **Required:** Yes
  - **Example:**
    ```json
    {
      "risk_tolerance": {"profit_taking": 50, "stop_loss": -500},
      "investment_horizon": 7,
      "liquidity_preference": {"adjust_on_price_out_of_range": true},
      "risk_aversion_threshold": 0.1,
      "user_status": "new_user"
    }
    ```

- **`pool_id`**
  - **Description:** The unique identifier of the liquidity pool (e.g., Uniswap V3 pool ID).
  - **Data Type:** `string`
  - **Required:** Yes
  - **Default Value:** `"0x4e68ccd3e89f51c3074ca5072bbac773960dfa36"`
  - **Example:** `"0x4e68ccd3e89f51c3074ca5072bbac773960dfa36"`

- **`ddpg_agent_path`**
  - **Description:** The file path to the trained DDPG agent.
  - **Data Type:** `string`
  - **Required:** Yes
  - **Default Value:** `"model_storage/ddpg/ddpg_1"`
  - **Example:** `"model_storage/ddpg/ddpg_agent_2024"`

- **`ppo_agent_path`**
  - **Description:** The file path to the trained PPO agent.
  - **Data Type:** `string`
  - **Required:** Yes
  - **Default Value:** `"model_storage/ppo/lstm_actor_critic_batch_norm"`
  - **Example:** `"model_storage/ppo/ppo_agent_2024"`

- **`date_str`**
  - **Description:** The date string used to reference specific data or state information for the inference.
  - **Data Type:** `string`
  - **Required:** No
  - **Default Value:** `"2024-05-05"`
  - **Example:** `"2024-08-01"`

### **7. Predict Action (`POST /api/v1/predict-action/`)**

- **`pool_id`**
  - **Description:** The unique identifier of the liquidity pool (e.g., Uniswap V3 pool ID).
  - **Data Type:** `string`
  - **Required:** Yes
  - **Default Value:** `"0x4e68ccd3e89f51c3074ca5072bbac773960dfa36"`
  - **Example:** `"0x4e68ccd3e89f51c3074ca5072bbac773960dfa36"`

- **`ddpg_agent_path`**
  - **Description:** The file path to the trained DDPG agent.
  - **Data Type:** `string`
  - **Required:** Yes
  - **Default Value:** `"model_storage/ddpg/ddpg_1"`
  - **Example:** `"model_storage/ddpg/ddpg_agent_2024"`

- **`ppo_agent_path`**
  - **Description:** The file path to the trained PPO agent.
  - **Data Type:** `string`
  - **Required:** Yes
  - **Default Value:** `"model_storage/ppo/lstm_actor_critic_batch_norm"`
  - **Example:** `"model_storage/ppo/ppo_agent_2024"`

- **`date_str`**
  - **Description:** The date string used to reference specific data or state information for predicting actions.
  - **Data Type:** `string`
  - **Required:** No
  - **Default Value:** `"2024-05-05"`
  - **Example:** `"2024-08-01"`

### **8. Download Log Files (`GET /api/v1/file/download/`)**

- **`file_path`**
  - **Description:** The relative path to the directory where the file is stored.
  - **Data Type:** `string`
  - **Required:** Yes
  - **Example:** `"model_outdir_csv/ddpg/ppo_prod1"`

- **`file_name`**
  - **Description:** The name of the file to download.
  - **Data Type:** `string`
  - **Required:** Yes
  - **Example:** `"train_logs.csv"`


1. **Authentication:**


2. **Rate Limiting:**


3. **Error Handling:**
   
4. **Versioning:**
   