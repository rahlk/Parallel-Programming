# heatmap for lake.cu
set loadpath "~/gnuplot/share/gnuplot/5.0"
set terminal postscript
set cbrange[-1:1]
set xrange[0:1]
set yrange[0:1]

set output 'lake_i.eps'
plot 'lake_i.dat' using 1:2:3 with image

set output 'lake_f.eps'
plot 'lake_f.dat' using 1:2:3 with image

set output 'lake_f_1.eps'
plot 'lake_f_1.dat' using 1:2:3 with image

set output 'lake_f_0.eps'
plot 'lake_f_0.dat' using 1:2:3 with image

set output 'lake_f_3.eps'
plot 'lake_f_3.dat' using 1:2:3 with image

set output 'lake_f_2.eps'
plot 'lake_f_2.dat' using 1:2:3 with image
