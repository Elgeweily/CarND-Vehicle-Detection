# Test find cars function
# Read test image
img = mpimg.imread('test_images/test1.jpg')
    
ystart = 400
ystop = 500
scale = 0.9
out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block)

ystart = 480
ystop = 650
scale = 1.5
out_img = find_cars(out_img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block)

plt.figure(figsize=(10,20))
plt.imshow(out_img)