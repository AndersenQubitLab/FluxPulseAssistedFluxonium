# Flux Pulse Assisted Readout of a Fluxonium Qubit

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Description

This project is composed of two main parts: readout and single qubit gates. 

In the readout portion of the project we investigate the dispersive shift landscape as a function of external flux bias of a superconducting fluxonium qubit capacitively coupled to a readout reasonator, where QuTiP is used to model the coupled system. Further, we can investigate the detunings of higher order transitions of the fluxonium with the readout resonator and the subsequent effects on the dispersive shift, as well as, the effects of changing energy parameters of the qubit and other defining parameters of the resonator and their coupling. Lastly in this section, we can simulate the readout dynamics of the qubit (with and without the addition of quasistatic 1/f flux noise) when a flux pulse with a finite ramp time is applied during the readout in order to tune the qubit to a bias point that possesses a higher dispersive shift by numerically solving the Langevin equation. We can also examine the effect of Purcell decay, relaxation due to dielectric loss, and flux pulse ramp time on the resulting readout error.

In the single qubit gate portion of the project, we provide optimization of a single qubit gate performed on a fluxonium qubit at the 'sweet-spot' with an accompanying DRAG pulse. Optimization is performed using the CMA global optimizer. Finally, we allow for the simulation of single qubit gate performance in the presence of quasistatic 1/f flux noise.

## Installation

### Prerequisites
- [Python 3.8.8+]
- [NumPy 1.16.6+]
- [SciPy 1.0+]
- [Cython 0.29.20+]
- [QuTiP 4.7.0]

### Cloning the Repository
Clone this repository to your local machine using [Git](https://github.com/AndersenQubitLab/FluxPulseAssistedFluxonium.git)

## Other
[Physical Review Applied publication](https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.22.014079)
[arXiv manuscript](https://arxiv.org/abs/2309.17286)
