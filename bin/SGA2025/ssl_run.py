from SSL_Match import ssl_match

for i in range(92):
    # Format the index number with leading zeros to read in each of the files
    filename = f"/pscratch/sd/i/ioannis/SGA2025/ssl/v1/input2/ssl-parent-chunk{i:03d}-v1.hdf5"
    ssl_match(filename)