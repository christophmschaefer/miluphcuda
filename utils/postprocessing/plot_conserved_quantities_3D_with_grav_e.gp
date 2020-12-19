#! /usr/bin/gnuplot -p

# Script to plot conserved quantities over time from conserved quantities logfile:
#     (1) SPH particle number
#     (2) total mass
#     (3) energy components
#     (4) linear momentum
#     (5) angular momentum
#     (6) barycenter position
#     (7) barycenter velocity

# NOTE: Works only for 3D simulations, and conserved quantities logfiles including grav. energy!


# file to plot
filename = "conserved_quantities.log"


set terminal postscript eps size 10.0,22.0 enhanced color font 'Helvetica,22' linewidth 1.5
set output "conserved_quantities.eps"


set lmargin at screen 0.14
set rmargin at screen 0.97

set key at graph 0.94,0.91 font ',27'

set multiplot layout 7,1

    set tmargin at screen 0.99
    set bmargin at screen 0.91
    set ylabel "Conserved quantities (SI units)" offset -2.0,-47.0 font ',30'
    plot filename u 1:2 w l title "SPH particle number"
    unset ylabel
    
    set tmargin at screen 0.89
    set bmargin at screen 0.81
    plot filename u 1:5 w l title "Total mass"
    
    set tmargin at screen 0.79
    set bmargin at screen 0.62
    plot filename u 1:6 w l title "Total kin. energy", \
         filename u 1:7 w l title "Total int. energy", \
         filename u 1:8 w l title "Total grav. energy", \
         filename u ($1):($6+$7+$8) w l lw 3.0 title "Total energy"
    
    set tmargin at screen 0.60
    set bmargin at screen 0.47
    plot filename u 1:9 w l lw 2.5 title "Lin. momentum (abs)", \
         filename u 1:10 w l title "Lin. momentum (x)", \
         filename u 1:11 w l title "Lin. momentum (y)", \
         filename u 1:12 w l title "Lin. momentum (z)"
    
    set tmargin at screen 0.45
    set bmargin at screen 0.32
    plot filename u 1:13 w l lw 2.5 title "Ang. momentum (abs)", \
         filename u 1:14 w l title "Ang. momentum (x)", \
         filename u 1:15 w l title "Ang. momentum (y)", \
         filename u 1:16 w l title "Ang. momentum (z)"
    
    set tmargin at screen 0.30
    set bmargin at screen 0.18
    plot filename u 1:17 w l title "Barycenter pos. (x)", \
         filename u 1:18 w l title "Barycenter pos. (y)", \
         filename u 1:19 w l title "Barycenter pos. (z)", \
         filename u ($1):(sqrt($17*$17+$18*$18+$19*$19)) w l lw 2.5 title "Barycenter pos. (abs)"
    
    set tmargin at screen 0.16
    set bmargin at screen 0.04
    set xlabel "Time (s)" offset 0.0,-1.0 font ',28'
    plot filename u 1:20 w l title "Barycenter vel. (x)", \
         filename u 1:21 w l title "Barycenter vel. (y)", \
         filename u 1:22 w l title "Barycenter vel. (z)", \
         filename u ($1):(sqrt($20*$20+$21*$21+$22*$22)) w l lw 2.5 title "Barycenter vel. (abs)"
    unset xlabel
    
unset multiplot

