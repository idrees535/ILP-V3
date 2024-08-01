## V3 Improvements
1. Integarte voyager simulator
2. Improve policy functions
3. Improve obs space, action space and reward function


### Prerequisites

- Linux/MacOS environment
- Python 3.9.19 (Please ensure this strictly, otherwise you will get into dependency conflict issues)
- NVM, Node.js, ganache and solc
- Ganache CLI v6.12.2 (ganache-core: 2.13.2)

Install NVM:
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
```
Note: Ensure you have curl installed. If not, install it with sudo apt install curl.
Replace v0.39.1 with the latest release available on the nvm GitHub page.
Restart your terminal or source your profile to ensure nvm commands are available:

```bash
source ~/.bashrc
```
Install Node.js using nvm:
```bash
nvm install 16.3.0
nvm use 16.3.0
```

Install Ganache CLI using npm:
```bash
npm install -g ganache-cli@7.9.1
```

Install solc-js via npm: 
```bash
npm install -g solc
```
### Installing Dependencies

1. Clone the repository:

```bash
git clone https://github.com/yourgithubusername/Intelligent-Liquidity-Provisioning-Framework-V2.git
```

2. Navigate into the cloned directory:

```bash
cd Intelligent-Liquidity-Provisioning-Framework-V2
```

3. Create and activate a virtual environment:

```bash
python3 -m venv ilp_venv
source ilp_venv/bin/activate
```

4. Install required Python packages:

```bash
pip install -r requirements_td.txt
```

5. Ensure you have the `solc`, `ganache`, and `nvm` installed and upto date

6. Open a new terminal and start ganache cli in this, please ensure that latest version of ganache is installed in your systems
```bash
ganache-cli --gasLimit 10000000000 --gasPrice 1 --hardfork istanbul
```

### Django App Deployment

1. In a new terminal Navigate to the Django project directory:

```bash
cd django_app
```

2. Apply migrations to initialize the database:

```bash
python manage.py migrate
```

3. Run the Django development server:

```bash
python manage.py runserver 0.0.0.0:8000
```

The server will start, and the API endpoints will be accessible locally.

