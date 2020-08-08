ddpg-scheduler

This project includes a Deep Reinforcement Learning based Scheduler for High Volume Flexible Time(HVFT) applications. The use case for this scheduler is large scale IoT networks that rely on high network performance. Scheduling is known to be a notoriously difficult problem and over the years there have been a lot of different machine learning approaches to solve it. My solution to it is based on Reinforcement Learning or most specifically on the Deep Deterministic Policy Gradient(DDPG) algorithm. The scheduler is very performant, trains very quickly and provides really good results. 

To do a test run of the scheduler, run the following commands:

- pip install -r requirements.txt
- python ddpg/run_ddpg.py
