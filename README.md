# ising core
 An implementation of the 2d Ising model in Julia. This model assumes a square lattice for simplicity. This program tracks and saves to files: total energy, magnetic susceptibility, absolute magnetization and specific heat; it also tracks total execution time.

 
 ## Running the simulation
 To use, run this script, with the help of the following optional arguments:

`--path`: you can continue the simulation with old data in the specified directory,
`--size`: length of the side of the 2d square lattice,
`--threshold`: number of Monte Carlo steps for the lattice to equilibrate the lattice,
`--mcs`: specify number of Monte Carlo steps for the simulation,
`--step-size`: how often the program should sample data (number of Monte Carlo steps inbetween),
`--temp-min`: specify lowest temperature in range,
`--temp-max`: specify highest temperature in range,
`--temp-step`: step in the temperature range.