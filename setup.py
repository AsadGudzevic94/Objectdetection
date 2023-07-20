from cx_Freeze import setup, Executable
import sys

setup(name='Simple object detection program',
      version='0.1',
      description = 'This program detects objects in real time',
      executables=[Executable('main.py')])