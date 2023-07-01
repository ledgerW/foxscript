# LLMWriter  


***  


# Getting Started  

## Install  
  

### Pull Repo  
`git clone https://github.com/ledgerW/llmwriter.git`

### Python  
1. Install Anaconda  

2. Create a virtual conda environment for this project:  
From project root: `conda env create -f environment.yml`  
Activate env: `activate llmwriter`

### Docker  
1. Install Docker  

### AWS CLI and Credentials  
1. Install AWS CLI  
2. Configure AWS .credentials and .config (see AWS Docs)  

### NodeJS / NPM  
1. Install NodeJS  
2. Install TS/JS dependencies  

**NextJS Frontend**
  - From `frontend`: `npm install`  

**Serverless Services**  
Install for each service individually.  
  - From `services/api`: `npm install`  
  - From `services/load_report`: `npm install`

### .env Files  
They are in several locations.  We can probably streamline this in this future.  
Populate `.env.template` and rename to `.env.local` in each of the following locations:  
- `.`: This one is for Weaviate (the vector database)
- `./frontend`: This one is for Next Frontend  
- `./services/api`: This one is for the API service (lambdas in AWS)  
- `./services/load_report`: This one is for a service that processes and loads threat intel reports into Weaviate
  

## Run Local Dev Environment  
***  
This includes four main components:
1. Run the local instance of Weaviate DB  
2. Run the local backend API server (this is an API Gateway / Lambda emulator)  
3. Run the local NextJS frontend  
4. Have API keys in .env.local files  
  - **OpenAI** and **Auth0** API services are used even in local development

### 1. Start Local Weaviate 
From project root: `python weaviate/start_weaviate.py --id [A BACKUP ID]`  
  - You'll need a backup id to load the schema and some dev data  
  - This uses docker compose to build and run a detached container  
  - Shut down with `python weaviate/stop_weaviate.py`  
  - There are backup and restore scripts too.  

### 2. Start Backend API Server
This runs the `services/api` service locally with Serverless Offline, so we
can make calls to our serverless backend like any local server, and while mimicking
the AWS APIGateway and Lambda environment.  
  
From `services/api`: `sls offline`  

### 3. Start NextJS Frontend Server  
From `frontend`: `npm run dev`



## DevOps  
***  

**CI/CD Services**  
[Vercel - Frontend](https://vercel.com/ledgerw/oxpecker)  
[Seed - Backend](https://console.seed.run/oxpecker/oxpecker)  

**Managed Cloud Database**  
[Weaviate Cloud](https://console.weaviate.cloud/dashboard)  

**Services**  
[GitHub]()  
[OpenAI]()  
[Serper](https://serper.dev/dashboard)
[Auth0](https://manage.auth0.com/dashboard/us/dev-0qj5tx1py4vgutlt/insights)  
[NewRelic](https://one.newrelic.com/dashboards/detail/Mzg5ODE2M3xWSVp8REFTSEJPQVJEfGRhOjMyNDAzNjM?account=3898163&state=ec3e76bf-3803-003e-f804-ac206f9a8f4e)  

### Deployment Workflow
[See Project Wiki](https://github.com/ledgerW/oxpecker/wiki/Dev-Workflow)



## Other  
***  

### Deploy Serverless Service From Terminal
This will be replaced ultimately with GitHub-triggered CI/CD
1. Activate your AWS credentials in each active terminal  
  - `set AWS_PROFILE=YOUR_PROFILE_NAME`  
2. from `services/[SERVICE]`: `sls deploy [args]`  

### Backup Locations (doesn't exist yet)  
S3: `oxpecker.data.dev/weaviate/backup`  

 







