# foxscript.  
[![Seed Status](https://api.seed.run/foxscript/foxscript/stages/prod/build_badge)](https://console.seed.run/foxscript/foxscript)

***  

# Getting Started  

## Install  
  

### Pull Repo  
`git clone https://github.com/ledgerW/foxscript.git`

### Python  
1. Install Anaconda  

2. Create a virtual conda environment for this project:  
From project root: `conda env create -f environment.yml`  
Activate env (Mac): `conda activate foxscript`
Activate env (Windows): `activate foxscript`

### Docker  
1. Install Docker  

### AWS CLI and Credentials  
1. Install AWS CLI  
2. Configure AWS .credentials and .config (see AWS Docs or ask Ledger)  

### NodeJS / NPM  / ServerlessFramework
1. Install NodeJS  
2. Install ServerlessFramework: `npm install -g serverless`

**Serverless Services**  
You would have to install for each service individually if deploying from your machine, but
to run locally with serverless offline we really only need to install packages for the `api` service.
Seed CI/CD handles the deployment on pushes/merges to `dev` and `prod` branches in Github.
  - From `services/api`: `npm install`  

### .env Files  
`services/api` and `services/data` expect env vars, but again at the moment we only run
serverless offline with the `api` service, so we only need to place the `.env` in the `services/api`
folder.  Ledger can share the `.env`.
- `./services/api`: This one is for the API service (lambdas in AWS)  
  

## Run Local Dev Environment  
### 1. Start Backend API Server
This runs the `services/api` service locally with Serverless Offline, so we
can make calls to our serverless backend like any local server, and while mimicking
the AWS APIGateway and Lambda environment.  

**Note:** currently, the backend assumes interaction with the Bubble.io frontend and db.
So, I have been testing and debugging the backend in conjunction with Bubble.io.  I can
share a notebook with example calls. I definitley plan to replace the notebook with a suite
of unit tests to be integrated with CI/CD (pretty soon; just haven't done it yet). 
  
From `services/api`: `sls offline`  



## DevOps and Service Dependencies
***  

**CI/CD Services**  
[Seed - Backend (AWS)](https://console.seed.run/foxscript/foxscript)  
[Bubble.io - Frontend](https://bubble.io/home/apps)  

**Managed Vector Database**  
[Weaviate Cloud](https://console.weaviate.cloud/dashboard)  

**Services**  
[Azure OpenAI](https://oai.azure.com/portal/49fa80b908e34bb0be87301438bb8fef/deployment?tenantid=2e635a9d-e281-40d5-ab79-68879819cbf7)  
[OpenAI](https://platform.openai.com/account/usage)  
[Serper](https://serper.dev/dashboard)  
[Postmark](https://account.postmarkapp.com/login)


***  

**Archived - the below is what local dev looked like in the very beginning**  

This includes four main components:
1. Run the local instance of Weaviate DB  
2. Run the local backend API server (this is an API Gateway / Lambda emulator)  
3. Run the local NextJS frontend  
4. Have API keys in .env.local files  
  - **OpenAI** and **Auth0** API services are used even in local development

### 2. Start Local Weaviate 
From project root: `python weaviate/start_weaviate.py --id [A BACKUP ID]`  
  - You'll need a backup id to load the schema and some dev data  
  - This uses docker compose to build and run a detached container  
  - Shut down with `python weaviate/stop_weaviate.py`  
  - There are backup and restore scripts too.  

### 3. Start NextJS Frontend Server  
From `frontend`: `npm run dev`  

 







