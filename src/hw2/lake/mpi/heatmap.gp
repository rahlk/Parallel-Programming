# heatmap for lake.cu

set terminal png

set xrange[0:0.5]
set yrange[0:0.5]

set output 'lake_1_i.png'
plot 'lake_node_1.dat' using 1:2:3 with image

set output 'lake_2_i.png'
plot 'lake_node_2.dat' using 1:2:3 with image

set output 'lake_3_i.png'
plot 'lake_node_3.dat' using 1:2:3 with image

set output 'lake_4_i.png'
plot 'lake_node_4.dat' using 1:2:3 with image
