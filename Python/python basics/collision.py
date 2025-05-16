def simulate_collisions(m2, v2_init):
    m1 = 1  # kg
    v1 = 0  # 1kg block is initially at rest
    v2 = v2_init

    collisions = 0

    while True:
        # Check if blocks will collide
        if v2 < v1:
            # Elastic collision between two blocks
            new_v1 = ((m1 - m2)*v1 + 2*m2*v2) / (m1 + m2)
            new_v2 = ((m2 - m1)*v2 + 2*m1*v1) / (m1 + m2)

            v1, v2 = new_v1, new_v2
            collisions += 1
        else:
            # 1kg block hits the wall
            if v1 < 0:
                v1 = -v1
                collisions += 1
            else:
                # If 1kg block is not going to hit the wall or collide anymore
                if v2 > v1:
                    break  # Big block is moving away, no more collisions

    return collisions


# User input
mass_of_second_block = float(input("Enter mass of the second block (in kg): "))
initial_velocity = float(input("Enter initial velocity of the second block towards the first (positive number): "))

total_collisions = simulate_collisions(mass_of_second_block, -abs(initial_velocity))
print(f"Total collisions: {total_collisions}")