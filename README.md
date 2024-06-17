## Global MMM 
MMM stands for Media Mix modelling. MMM description

## Application Architecture 
![Architecture](./Global-MMM-architecture.png)

To read more about Global MMM architecture . Go to this link : [Architecture explanation](./architecture-explantation.md) 

## Tech Stack and dependencies 
* Angular - Tested with 17.1.1
* Docker - Tested with 26.1.1
* flask - Tested with 3.0.2

## Running Global MMM application in local environment
- Clone the github repository : `https://github.com/Brainlabs-Digital/global-mmm-v2.git`
- put the service account file `global-mso-ai-use-case-a4f897442c64` and `config.ini` in these folders of backend : 
    * Robyn
    * pymc
    * main-api-global-mmm
- Run the command in CLI (`./backend`) : `docker-compose up -d`
- Run the command in CLI (`./web`) : `ng serve`
- Install the necessary libraries in frontend : `npm install`
- your application is come at `http://localhost:4200`

## Deployment process of Global MMM application 


## For Git best practices 
Go to this Link [Git best practices](./best-practices.md) 

