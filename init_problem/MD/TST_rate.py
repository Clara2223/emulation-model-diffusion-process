
#Calculating the rate constant of a chemical reaction using Transition State Theory (TST) 
##---------------------------------------------------------------------


##---------------------------------------------------------------------
#1 : Harmonic approximation of the potential energy surface (PES)
##---------------------------------------------------------------------
# this is only a harmonic approx. since the well is gaussian i.e. not quadratic.
#I calculate the gibbs free energy with thermodynamic relation

#hessian matrix construction
# - the second derivative of the potential energy V wrt. the coordinates x & y;
Hxy=delta**2 V/delta x delta y

#freq. determination
# - the eigenvalues of the hessian matrix are the vibrational frequencies of the system
omega_x, omega_y = sqrt(eigenvalues(Hxy)/m)
nu_x, nu_y = omega_x/(2 pi), omega_y/(2 pi)

#vibrational partition function
q_vib= product_i [ exp(-h nu_i / 2 k_B T) /(1-exp(-h nu_i/kT)) ]
#then these vibrations are used to express the entropy with the relation S=k_B ln(q_vib)+k_B T (delta ln(q_vib)/delta T)
S=k_B ln(q_vib)+k_B T (delta ln(q_vib)/delta T)
# the enthalpy can be expressed /approximated by the potential energy at the minimum of the well, i.e. H=V_min difference to the transition state, which is the barrier height equivalent energy in the PES.
# + the the zero point correction ?
H=V_min - V_TS + ?1/2 h (nu_x + nu_y)?
# T I defined for the MD sim.
G=delta_H - T delta_S

#Lastly the Eyrin TST rate constant:
k_TST = (k_B T / h) exp(-delta_G / k_B T)    

##---------------------------------------------------------------------
#2 : Probability density & MD sampling 
##---------------------------------------------------------------------
# using my MD sampling, one obtaisn the more 'correct' free energy surface, which is not harmonic.

#potential of mean force (PMF) / Free energy profile
# P(x) probability density of the system along the reaction coordinate x (which is obtained from the MD sampling) to free energy G(x) with the relation:
G(x)=-k_BT ln(P(x)) + C

#free energy difference: comparing prob. at TS point against the integrated prob. density over the reactant well
Delta_G=-k_BT*ln(P(x_TS)/integral_well(P(x)dx))

#rate constant from prob. int: rate const. prop. to ratio of the prob. at TS to the integrated prob. density over the reactant well - often coupled w. avg. velocity crossing terms in dynamical corrections
k=v_x_abs_avg * P(x_TS)/integral_well(P(x)dx)


##---------------------------------------------------------------------
#3 : very rough approx: Using just the energy difference TS-V_min as E_act in Arrhenius
##---------------------------------------------------------------------
# this would be the approximation that the entropy does not matter / is negligable
delta_E=V_min-V_TST
k_tst= (k_B T / h) exp(-delta_E / k_B T)



#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#then for a poisson process (like a rare-event transition), the expected waiting time for the first transition is the inverse of the rate:
T_TS=1/k

expected_transitions== N = k * T

total production time of MD simulation= (n_steps - burn_in) * dt
safe_production_time = 2 * (n_steps - burn_in) * dt
#        meaning it calculates the double to be sure we have the requested one rate state transition
# because: MD is random - statistical fluctuations may cause less transitions that hoping for otherwise.
