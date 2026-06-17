import numpy as np

def test_overlap():
    pixscale = 0.262
    
    bx1, by1 = 138.94322, 155.5274
    sma1 = 0.4085466 * 60 / 2 / pixscale  # 46.8 pixels
    ba1, pa1 = 0.45156658, 140.08327
    semib1 = sma1 * ba1  # 21.1 pixels
    
    bx2, by2 = 104.348885, 96.19282
    sma2 = 0.5139345 * 60 / 2 / pixscale  # 58.9 pixels
    ba2, pa2 = 0.39755225, 132.46239
    semib2 = sma2 * ba2  # 23.4 pixels
    
    # Check 1: Is center of ellipse 1 inside ellipse 2?
    theta2 = np.deg2rad(pa2)
    dx = bx1 - bx2
    dy = by1 - by2
    xp =  dx * np.sin(theta2) + dy * np.cos(theta2)
    yp = -dx * np.cos(theta2) + dy * np.sin(theta2)
    inside1 = (xp/sma2)**2 + (yp/semib2)**2 <= 1
    print(f"Center 1 in ellipse 2: {inside1}, dist = {np.sqrt((xp/sma2)**2 + (yp/semib2)**2):.2f}")
    
    # Check 2: Is center of ellipse 2 inside ellipse 1?
    theta1 = np.deg2rad(pa1)
    dx = bx2 - bx1
    dy = by2 - by1
    xp =  dx * np.sin(theta1) + dy * np.cos(theta1)
    yp = -dx * np.cos(theta1) + dy * np.sin(theta1)
    inside2 = (xp/sma1)**2 + (yp/semib1)**2 <= 1
    print(f"Center 2 in ellipse 1: {inside2}, dist = {np.sqrt((xp/sma1)**2 + (yp/semib1)**2):.2f}")
    
    # Generate boundary points of ellipse 1
    t = np.linspace(0, 2*np.pi, 64, endpoint=False)
    xp_local = sma1 * np.cos(t)
    yp_local = semib1 * np.sin(t)
    
    # Transform to image coords
    dx_boundary = xp_local * np.sin(theta1) - yp_local * np.cos(theta1)
    dy_boundary = xp_local * np.cos(theta1) + yp_local * np.sin(theta1)
    x1 = bx1 + dx_boundary
    y1 = by1 + dy_boundary
    
    # Test these points against ellipse 2
    dx_test = x1 - bx2
    dy_test = y1 - by2
    xp_test =  dx_test * np.sin(theta2) + dy_test * np.cos(theta2)
    yp_test = -dx_test * np.cos(theta2) + dy_test * np.sin(theta2)
    inside = (xp_test/sma2)**2 + (yp_test/semib2)**2 <= 1
    print(f"Any boundary 1 inside ellipse 2: {np.any(inside)}")
    print(f"Min distance: {np.min(np.sqrt((xp_test/sma2)**2 + (yp_test/semib2)**2)):.2f}")

def test_overlap_full():
    pixscale = 0.262

    bx1, by1 = 138.94322, 155.5274
    sma1 = 0.4085466 * 60 / 2 / pixscale
    ba1, pa1 = 0.45156658, 140.08327
    semib1 = sma1 * ba1

    bx2, by2 = 104.348885, 96.19282
    sma2 = 0.5139345 * 60 / 2 / pixscale
    ba2, pa2 = 0.39755225, 132.46239
    semib2 = sma2 * ba2

    theta1 = np.deg2rad(pa1)
    theta2 = np.deg2rad(pa2)

    # Also test boundary of ellipse 2 against ellipse 1
    t = np.linspace(0, 2*np.pi, 64, endpoint=False)
    xp_local2 = sma2 * np.cos(t)
    yp_local2 = semib2 * np.sin(t)

    dx_boundary2 = xp_local2 * np.sin(theta2) - yp_local2 * np.cos(theta2)
    dy_boundary2 = xp_local2 * np.cos(theta2) + yp_local2 * np.sin(theta2)
    x2 = bx2 + dx_boundary2
    y2 = by2 + dy_boundary2

    dx_test = x2 - bx1
    dy_test = y2 - by1
    xp_test =  dx_test * np.sin(theta1) + dy_test * np.cos(theta1)
    yp_test = -dx_test * np.cos(theta1) + dy_test * np.sin(theta1)
    dist = np.sqrt((xp_test/sma1)**2 + (yp_test/semib1)**2)
    print(f"Any boundary 2 inside ellipse 1: {np.any(dist <= 1)}")
    print(f"Min distance: {np.min(dist):.2f}")

    # Try with more samples
    t_dense = np.linspace(0, 2*np.pi, 256, endpoint=False)
    for ell_idx, (bx, by, sma, semib, theta, other_bx, other_by, other_sma, other_semib, other_theta) in enumerate([
        (bx1, by1, sma1, semib1, theta1, bx2, by2, sma2, semib2, theta2),
        (bx2, by2, sma2, semib2, theta2, bx1, by1, sma1, semib1, theta1)
    ]):
        xp_local = sma * np.cos(t_dense)
        yp_local = semib * np.sin(t_dense)
        dx_b = xp_local * np.sin(theta) - yp_local * np.cos(theta)
        dy_b = xp_local * np.cos(theta) + yp_local * np.sin(theta)
        x_b = bx + dx_b
        y_b = by + dy_b

        dx_t = x_b - other_bx
        dy_t = y_b - other_by
        xp_t =  dx_t * np.sin(other_theta) + dy_t * np.cos(other_theta)
        yp_t = -dx_t * np.cos(other_theta) + dy_t * np.sin(other_theta)
        dist = np.sqrt((xp_t/other_sma)**2 + (yp_t/other_semib)**2)
        print(f"Ellipse {ell_idx+1} boundary (256 pts) min dist: {np.min(dist):.3f}")

test_overlap_full()

#test_overlap()
