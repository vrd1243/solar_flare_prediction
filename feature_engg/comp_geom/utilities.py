#!/usr/bin/python
import os

def line_count_in_file(filepath): 
    
    if not os.path.exists(filepath):
        print("Path doesnt exist");
        return 0;
	          
    fhandle = open(filepath, 'r');
    
    count = 0; 
    for line in fhandle:
        count += 1;

    fhandle.close(); 

    return count;
