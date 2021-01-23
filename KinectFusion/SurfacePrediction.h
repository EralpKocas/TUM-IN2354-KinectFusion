


class SurfacePrediction {

    SurfacePrediction() {}

    // TODO: define a function to calculate raycast of a pixel

    // TODO: apply marching steps for per pixel u from minimum depth until finding a surface
        // stop conditions:
            // 1. when zero crossing is found
            // 2. -ve to +ve -> back-face is found
            // 3. if exceeds the working volume
            // 2 and 3 result as non-surface measurement at pixel u
            // TODO: implement conditions in a loop to understand continue to next pixel or not.
            // TODO: if not, calculate necessary vertex and normal maps.
        // for points very close to surface interface where F_{k}(p) = 0
            // surface normal for pixel u along which p -> compute directly using numerical derivative of F_{k}, SDF.
                // TODO: equation is on paper (Eq. 14)

        // TODO: understand and decide how to implement min and max sensor range (in paper -> [0.4, 8] meters)
        // TODO: understand the ray skipping
            // so far: decide on a step size to skip ray, maximum < μ.
                // around F(p) = 0 -> good approximation to true signed distance.

        // TODO: obtain higher quality intersections around the found intersections of SDF.
        // TODO: predicted vertex and normal maps are computed at interpolated location in the global frame.
};