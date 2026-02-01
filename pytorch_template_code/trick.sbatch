#!/bin/bash                                                                                                                                         
 
#SBATCH --job-name=interactive                                                                                                                       
#SBATCH --nodes=1                                                                                                                            
#SBATCH --partition=gpu_prod_long                                                                                                                   
#SBATCH --time=12:00:00                                                                                                                              
#SBATCH --exclude=sh[00,10-19]                                                                                                                      
 
echo "hello world"
while true
do
  sleep 1000
done

