#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021-05-06

Simple script to calculate the quantum--classical tunneling 
crossover.

@author: jkshenton
"""

from math import pi





# speed of light
c = 299792458.0 # m/s
hbar = 1.0545718001391127e-34 #m^2 kg s^-1
kB = 1.38064852e-23 # m^2 kg s^-2 K^-1

def calc_temperature(freq):

    T = freq * 1e2 * c * hbar / kB

    return T


if __name__ == "__main__":
    import argparse
    # Create the parser and add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='freq', help="Frequency in cm^-1", type=float)

    # Parse and print the results
    args = parser.parse_args()

    freq = args.freq
    
    T = calc_temperature(freq)
    print(f"Crossover temperature: {T:16.8f} K")
