#!/usr/bin/env python
# coding: utf-8

from scipy.optimize import minimize
from numpy import log as ln
import numpy as np



def obj_fun(x):
    beta = [101.71, 103.8, 78.06, 122.04]
    #alpha = [0.23, 0.65]
    
    ##select the active links here
    ## 1100 -> links 1 & 2 active
    ## 1010 -> Links 1 & 3 active
    ## 1001 -> Links 1 & 4 active
    ##..........................
    ##..........................
    ## 0111 -> Links 2, 3 & 4 active
    ## 1111 -> Links 1, 2, 3 & 4 active
    
    link = "1111"
    
    #the variables
    x1  = x[0]
    x2  = x[1]
    x3  = x[2]
    x4  = x[3]
    x12 = x[4]
    x13 = x[5]
    x14 = x[6]
    x23 = x[7]
    x24 = x[8]
    x34 = x[9]
    x123 = x[10]
    x124 = x[11]
    x134 = x[12]
    x234 = x[13]
    x1234 = x[14]
    
    # the parameter alphai_m the factor by which the throughput
    # of link i drops when links in set m are active.
    # m belongs to power set of {1,2,3,4} excluding the null sets. 
    alpha1_12 = 0.50
    alpha1_13 = 0.84
    alpha1_14 = 0.97
    alpha2_12 = 0.52
    alpha2_23 = 0.29
    alpha2_24 = 0.65
    alpha3_13 = 0.66
    alpha3_23 = 0.37
    alpha3_34 = 0.33
    alpha4_14 = 1.00
    alpha4_24 = 0.99
    alpha4_34 = 0.65
    
    alpha1_123 = 0.47
    alpha1_124 = 0.53
    alpha1_134 = 0.93
    alpha2_123 = 0.12 
    alpha2_124 = 0.32
    alpha2_234 = 0.52
    alpha3_123 = 0.51
    alpha3_134 = 0.22
    alpha3_234 = 0.12
    alpha4_124 = 1.00
    alpha4_134 = 0.66
    alpha4_234 = 0.69
    
    alpha1_1234 = 0.52
    alpha2_1234 = 0.26
    alpha3_1234 = 0.17
    alpha4_1234 = 0.67
    
    #isolated throughput
    beta1 = beta[0]
    beta2 = beta[1]
    beta3 = beta[2]
    beta4 = beta[3] 

    #returns the objective function as per the elected link to the scipy solver   
    if link == "1100":#links 1 & 2 active
        return -(ln(x1*beta1 + x12*alpha1_12*beta1) + ln(x2*beta2 + x12*alpha2_12*beta2))
    
    elif link == "1010":#links 1 & 3 active
        return -(ln(x1*beta1 + x13*alpha1_13*beta1) + ln(x3*beta3 + x13*alpha3_13*beta3))
    
    elif link == "1001":#links 1 & 4 active
        return -(ln(x1*beta1 + x14*alpha1_14*beta1) + ln(x4*beta4 + x14*alpha4_14*beta4))
    
    elif link == "0110":#links 2 & 3 active
        return -(ln(x2*beta2 + x23*alpha2_23*beta2) + ln(x3*beta3 + x23*alpha3_23*beta3))
    
    elif link == "0101":#links 2 & 4 active
        return -(ln(x2*beta2 + x24*alpha2_24*beta2) + ln(x4*beta4 + x24*alpha4_24*beta4))
    
    elif link == "0011":#links 3 & 4 active
        return -(ln(x3*beta3 + x34*alpha3_34*beta3) + ln(x4*beta4 + x34*alpha4_34*beta4)) 
    
    elif link == "1110":#links 1, 2 & 3 active
        return -(ln(x1*beta1 + x12*alpha1_12*beta1 + x13*alpha1_13*beta1 + x123*alpha1_123*beta1)
               + ln(x2*beta2 + x12*alpha2_12*beta2 + x23*alpha2_23*beta2 + x123*alpha2_123*beta2)
               + ln(x3*beta3 + x13*alpha3_13*beta3 + x23*alpha3_23*beta3 + x123*alpha3_123*beta3))
    
    elif link == "1101":# links 1, 2 & 4 active
        return -(ln(x1*beta1 + x12*alpha1_12*beta1 + x14*alpha1_14*beta1 + x124*alpha1_124*beta1)
               + ln(x2*beta2 + x12*alpha2_12*beta2 + x24*alpha2_24*beta2 + x124*alpha2_124*beta2)
               + ln(x4*beta4 + x14*alpha4_14*beta4 + x24*alpha4_24*beta4 + x124*alpha4_124*beta4))
    
    elif link == "1011": #links 1, 3 & 4 active        
        return -(ln(x1*beta1 + x13*alpha1_13*beta1 + x14*alpha1_14*beta1 + x134*alpha1_134*beta1)
               + ln(x3*beta3 + x13*alpha3_13*beta3 + x34*alpha3_34*beta3 + x134*alpha3_134*beta3)
               + ln(x4*beta4 + x14*alpha4_14*beta4 + x34*alpha4_34*beta4 + x134*alpha4_134*beta4))
    
    elif link == "0111": #links 1, 3 & 4 active
        return -(ln(x2*beta2 + x23*alpha2_23*beta2 + x24*alpha2_24*beta2 + x234*alpha2_234*beta2)
               + ln(x3*beta3 + x23*alpha3_23*beta3 + x34*alpha3_34*beta3 + x234*alpha3_234*beta3)
               + ln(x4*beta4 + x24*alpha4_24*beta4 + x34*alpha4_34*beta4 + x234*alpha4_234*beta4))
        
    elif link == "1111": #links 1, 2, 3 & 4 active
        return -(ln(x1*beta1 + x12*alpha1_12*beta1 + x13*alpha1_13*beta1 + x14*alpha1_14*beta1 + 
                x123*alpha1_123*beta1 + x124*alpha1_124*beta1 + x134*alpha1_134*beta1 + x1234*alpha1_1234*beta1)
             + ln(x2*beta2 + x12*alpha2_12*beta2 + x23*alpha2_23*beta2 + x24*alpha2_24*beta2 + 
                x123*alpha2_123*beta2 + x124*alpha2_124*beta2 + x234*alpha2_234*beta2 + x1234*alpha2_1234*beta2)
             + ln(x3*beta3 + x13*alpha3_13*beta3 + x23*alpha3_23*beta3 + x34*alpha3_34*beta3 + 
                x123*alpha3_123*beta3 + x134*alpha3_134*beta3 + x234*alpha3_234*beta3 + x1234*alpha3_1234*beta3)
             + ln(x4*beta4 + x14*alpha4_14*beta4 + x24*alpha4_24*beta4 + x34*alpha4_34*beta4 + 
                x124*alpha4_124*beta4 + x124*alpha4_124*beta4 + x234*alpha4_234*beta4 + x1234*alpha4_1234*beta4))
    
def constraint(x):
    # sum x_i = 1
    return sum(x) - 1


#initialize the variables with random values
x0 = np.random.random_sample(size = 15)
x0 /= x0.sum()
#print(obj_fun(x0))

#bounds for variables
b = [0.0,1.0]
bound = [b]*15

#add the constraint to solver
cons = {'type':'eq','fun': constraint}

#obtain the objective function & passed to scipy solver
sol = minimize(obj_fun,x0,method='SLSQP',bounds=bound, constraints=cons)


# print the vector x, containing optimal values of x_i
print(sol)
