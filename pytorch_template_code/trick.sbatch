#!/bin/bash                                                                                                                                         
 
#SBATCH --job-name=interactive                                                                                                                       
#SBATCH --nodes=1                                                                                                                            
#SBATCH --partition=gpu_prod_night                                                                                                                   
#SBATCH --time=12:00:00    
#SBATCH --exclude=sh[00,01,05,03,09,10-19,20-22,08]                                                                                                                                                                                                                                                
 
echo "hello world"
while true
do
  sleep 1000
done

