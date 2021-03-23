# Udacity Deep Reinforcement Learning Projects
This repository contains my solutions for homework given in the nanodegree.

### List of Projects
1.  **Collect Bananas or ( Navigation ) :** Train an agent to navigate a large world and collect yellow bananas, while avoiding blue bananas.


2.  **Reacher or ( Continuous Control ) :** a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal the agent is to maintain its position at the target location for as many time steps as possible.


3.  **Tennis or ( Collaboration and Competition ):** Two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.



# Installation 
	- Create conda env for `python3.6`
	```
		conda create -n Python3.6Test python=3.6    # install conda env for python3.6
		pip3 install ipykernel
		python -m ipykernel install --name Python3.6Test  # create a python3.6 kernal 
		conda activate Python3.6Test
	```
	- Install the unity application from Unity Hub (click here)[https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md] 
	- Install unity python dependencies
	```
		pip3 install unityagents
		pip3 install mlagents
	```

# Running in my mac
Activate ENV
`conda activate Python3.6Test`
Run Jupyter Notebook
`/Users/amitverma/opt/anaconda3/bin/jupyter_mac.command ; exit;`