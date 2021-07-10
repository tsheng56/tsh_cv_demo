import os

cmd = 'find ./ -name "*" > temp_lines'
os.system(cmd)
lines = open('temp_lines', 'r').readlines()
os.remove('temp_lines')

for line in lines:
    line = line.strip()
    if '__pycache__' in line:
        os.system(f'rm -rf {line}')
        print(f'rm -rf {line}')
        
        