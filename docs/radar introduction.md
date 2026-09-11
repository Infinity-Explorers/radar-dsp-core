our project is about **FMCW radar** which translates to (frequency modulated continues wave)
lets break this down a bit.

### radar principal
the main principal behind this is **wave propagation**

radar waves travel thro space and when they hit an object they either break into small waves that travel in random directions which is known as scattering or they just mean a reflective surface which just changes the wave direction.

the way this works is by receiving those waves after they hit something and measure the time $t$ between transmitting and receiving the waves. 

as the speed of the wave is known which equals the speed of light.

$$c= 3 \times 10^8 \quad m/s$$

the distance between the wave source (radar) and the target (surface the waves hit) can be calculated using the formula.

$$d= \frac{c \cdot t}{2}$$

### continues wave (CW)

the main idea behind the pulser radar (the simplest) was to send the wave and wait for the antenna to receive it then calculate the time and the distance using the previous formulas.

but this method required a lot of waiting for the signal to return so this causes a drop in the average power compared to the peak which results in the need of huge power peaks to make this come to life. 

the solution to this was the continues wave where u keep transmitting the signal all the time.

### frequency modulated (FM)

this is a good idea yet this wont work as u cant differentiate between the received signal and the transmitted as they both have the same frequency as shown in the figure.

![](../docs/assets/Screenshot%202026-08-09%20005842.png)

this is where we modulate the signal by changing the frequency over time instead of using constant frequency like we did previously.

the most common frequency modulation method is the sawtooth wave.
![](./assets/Screenshot%202026-08-09%20010455.png)
![](./assets/Screenshot%202026-08-09%20011543.png)

as u can see the difference between the transmitted signal and the received one became very obvious.

By varying frequency over time: 
* The transmitted signal is a continuous frequency sweep called a **chirp**. 
* The received echo is a time-delayed, frequency-shifted replica of the transmitted chirp.
* Mixing the transmitted and received signals generates a **beat frequency** ($f_b$). 
* The **beat frequency ($f_b$)** represents the instantaneous frequency difference between the transmitted signal and received echo, which is directly proportional to the target distance           ($f_b = S \cdot t$).

Key parameters from this modulation structure include: 
* **Bandwidth ($B$):** The total frequency sweep range (determines range resolution).
* **Chirp Duration ($T_c$):** The time taken to perform a single sweep. 
* **Chirp Slope ($S = \frac{B}{T_c}$):** The rate of frequency change over time. 

These parameters form the mathematical foundation for calculating distance, relative velocity, and arrival angle in our DSP pipeline.