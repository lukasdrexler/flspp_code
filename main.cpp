#include "main.h"



//======================================================================================================================

double get_cost(std::vector<double> point_distances){
    double cost = 0;
    for (int i = 0; i < point_distances.size(); ++i) {
        cost += point_distances[i];
    }
    return cost;
}


double euclidean_distance_squared(std::vector<double> &x, std::vector<double> &y){
    double dist = 0;
    for (int i = 0; i < x.size(); ++i) {
        dist += (x[i] - y[i])*(x[i] - y[i]);
    }
    return dist;
}


/*
void update_distances(std::vector<std::vector<double>> &P,
                                      std::vector<std::vector<double>> &centers,
                                      std::vector<int> &labels){
*/
/* Naive implementation (change later, such that only changed distances are updated)
 * Input: vector of points P, vector of (new) centers 'centers', list of indices of closest centers (i.e. cluster membership)
 *
 * Output: nada


}
*/


//VERSION WHERE POINTS ARE ACTUALLY OBJECTS OF CLASS "POINT"

void update_labels(std::vector<Point> &Points,
                   std::vector<Point> &centers,
                   std::vector<double> &point_distances,
                   std::vector<int> labels,
                   std::vector<int> second_closest){

    double min_dist, second_min_dist;
    int min_label, second_closest_label;

    for (int i = 0; i < Points.size(); ++i) {
        min_dist = std::numeric_limits<double>::max();
        second_min_dist = std::numeric_limits<double>::max();
        min_label = -1;
        second_closest_label = -1;

        for (int j = 0; j < centers.size(); ++j) {
            double dist = euclidean_distance_squared(Points[i], centers[j]);
            if(dist < min_dist){
                min_dist = dist;
                min_label = j;
            }
        }
        point_distances[i] = min_dist;
        labels[i] = min_label;
    }
}

//VERSION WHERE POINTS ARE VECS OF VECS OF DOUBLES
std::vector<int> update_labels(std::vector<std::vector<double>> &P,
                               std::vector<std::vector<double>> &centers,
                               std::vector<int> labels,
                               std::vector<double> &point_distances){

    double min_dist;
    int min_label;

    for (int i = 0; i < P.size(); ++i) {
        min_dist = std::numeric_limits<double>::max();
        min_label = -1;

        for (int j = 0; j < centers.size(); ++j) {
            double dist = euclidean_distance_squared(P[i], centers[j]);
            if(dist < min_dist){
                min_dist = dist;
                min_label = j;
            }
        }
        point_distances[i] = min_dist;
        labels[i] = min_label;
    }
    return labels;
}

//VERSION WHERE POINTS ARE ACTUALLY OBJECTS OF CLASS "POINT"
void update_labels_init(std::vector<Point> &Points,
                        std::vector<Point> &centers,
                        std::vector<double> &closest_center_distances,
                        std::vector<double> &second_closest_center_distances,
                        std::vector<double> &cumsums,
                        std::vector<int> &labels,
                        std::vector<int> &second_closest_labels){
    double min_dist;
    int min_label;

    for (int i = 0; i < Points.size(); ++i) {
        //min_dist = std::numeric_limits<double>::max();
        min_label = -1;

        for (int j = 0; j < centers.size(); ++j) {
            double dist = euclidean_distance_squared(Points[i], centers[j]);

            //if center j is closer to point i than center(i)
            if(dist < closest_center_distances[i]){
                //former closest center of i becomes second-closest center now
                second_closest_labels[i] = labels[i];
                second_closest_center_distances[i] = closest_center_distances[i];

                //j becomes new center of i
                closest_center_distances[i] = dist;
                labels[i] = j;
            }
            else if(dist < second_closest_center_distances[i] && labels[i] != j){
                //closest center stays the same but second closest has to be updated
                second_closest_center_distances[i] = dist;
                second_closest_labels[i] = j;
            }
        }


        if(i==0){
            cumsums[i] = closest_center_distances[i];
        }
        else{
            cumsums[i] = cumsums[i-1]+closest_center_distances[i];
        }
    }
}



// VERSION WHERE POINTS ARE VECS OF VECS OF DOUBLE
void update_labels_init(std::vector<std::vector<double>> &P,
                        std::vector<std::vector<double>> &centers,
                        std::vector<int> &labels,
                        std::vector<double> &point_distances,
                        std::vector<double> &cumsums){
    double min_dist;
    int min_label;

    for (int i = 0; i < P.size(); ++i) {
        min_dist = std::numeric_limits<double>::max();
        min_label = -1;

        for (int j = 0; j < centers.size(); ++j) {
            double dist = euclidean_distance_squared(P[i], centers[j]);
            if(dist < min_dist){
                min_dist = dist;
                min_label = j;
            }
        }

        point_distances[i] = min_dist;
        labels[i] = min_label;

        if(i==0){
            cumsums[i] = min_dist;
        }
        else{
            cumsums[i] = cumsums[i-1]+min_dist;
        }
    }
}

std::vector<double> cumulative_sums(std::vector<double> &point_distances){
    /*
     * Helper function for the D²-sampling (same method as in sklearn (good?))
     *
     * Input: vector of size |P| that contains for each point the current distance to its closest center, i.e. its
     * "contribution" to the overall cost of the current solution.
     *
     * Output: vector of size |P| where entry [i] contains the sum of the distances of points [0, ... , i-1] (cumulative
     * sum).
     */

    std::vector<double> cumsums;
    double sum = 0;
    for (std::vector<double>::iterator it = point_distances.begin(); it != point_distances.end(); ++it) {
        sum += *it;
        cumsums.push_back(sum);
    }
    return cumsums;
}


int choose_center(std::vector<double> &point_distances, double pot){
    /*
     * Input: vector of cumulative sums of cost, cost ("potential") of current solution
     *
     * Output: index of a point, s.t. each point has probability of being chosen according to D²-sampling
     */

    std::vector<double> cumsums = cumulative_sums(point_distances); //kann weg

    //create random integer in range [0, ... ,pot]

    double randnr = rand()*pot/RAND_MAX;


    for (int i = 0; i < cumsums.size(); ++i) {
        if(randnr < cumsums[i]){
            return i;
        }
    }
    std::cout << "If this gets printed, the generated number was too big!";
    return cumsums.size()-1; //failsafe, probably not needed
}


std::vector<std::vector<double>> update_centroids(std::vector<std::vector<double>> &P,
                                                  std::vector<int> labels,
                                                  std::vector<std::vector<double>> centers){


    int dim = P[0].size();      //shorthand for number of coordinates
    std::vector<std::vector<double>> centroids;     //empty vector that gets filled with new centroids
    std::vector<int> cluster_sizes(centers.size());     //vector that will contain for each center (cluster) the number of points with that label
    std::vector<double> init (dim);     //just the dim-dimensional 0-vector

    // initialize vector of k copies of [0,...,0] (which will successively be updated to become the centroids)
    /*  Note that the centroid is the "mean" of a cluster, i.e. sum of all points in the cluster divided by number.
        We do not know cluster size in advance, so we first add up all points in each cluster and keep track of how many
        points were added in which cluster. In the end, every sum of points gets divided by the number of points in the
        respective cluster.
    */

    for (int i = 0; i < centers.size(); ++i) {
        centroids.push_back(init);
    }

    //iterate over all points
    for (int i = 0; i < P.size(); ++i) {
        int label = labels [i];     //check to which cluster the current point belongs

        for (int j = 0; j < dim; ++j) {
            centroids[label][j] += P[i][j];     //add coordinates of current point to coord. of centroid-to-be
        }
        cluster_sizes[label] += 1;  //increase size couter by one, as one point has been added
    }

    for (int i = 0; i < centroids.size(); ++i) {
        for (int j = 0; j < dim; ++j) {
            centroids[i][j] = centroids[i][j]/cluster_sizes[i]; //divide each coordinate of the summed up vectors by number of points in resp. cluster
        }

    }

    return centroids;
}


//=================PROBABLY NOT NEEDED, AS CURRENTLY CLUSTERS ARE NOT STORED AS VECTORS=================================

std::vector<double> get_centroid(std::vector<std::vector<double>> &C){
/*
 * Input: vector of points, representing a cluster (should probably be changed to vector of just the indices
 *
 * Output: centroid (aka "average") of those points
 */
    int dim = C[0].size();
    std::vector<double> centroid(dim);

    for(std::vector<std::vector<double>>::iterator it = C.begin(); it != C.end(); ++it) {
        std::vector<double> x = *it;
        centroid[0] += x[0]/C.size();
        centroid[1] += x[1]/C.size();
    }
    return centroid;
}

//======================================================================================================================


int get_closest_center(std::vector<double> &p, std::vector<std::vector<double>> &centers){
    /*
     * Input: point p and vector of centers
     *
     * Output: index of center closest to p
     */

    int index = -1;
    double min_dist = std::numeric_limits<double>::max();
    for (int i = 0; i < centers.size(); ++i) {
        double current_dist = euclidean_distance_squared(p,centers[i]);
        if(current_dist < min_dist){
            index = i;
            min_dist = current_dist;
        }
    }
    return index;
}


std::vector<Point> kmeans_pp(std::vector<Point> &Points,
                             std::vector<double> &closest_center_distances,
                             std::vector<double> &second_closest_center_distances,
                             std::vector<double> &cumsums,
                             std::vector<int> &labels,
                             std::vector<int> &second_closest_labels,
                             int k){

    std::vector<Point> centers;

    //choose first center uniformly at random
    double randnr = rand()*Points.size()/RAND_MAX;
    int randint = ((int) randnr);
    centers.push_back(Points[randint]);

    while(centers.size() < k) {


        update_labels_init(Points, centers, closest_center_distances, second_closest_center_distances, cumsums, labels, second_closest_labels);

        int new_center = choose_center(closest_center_distances, cumsums.back());

        centers.push_back(Points[new_center]);
    }

    update_labels_init(Points, centers, closest_center_distances, second_closest_center_distances, cumsums, labels, second_closest_labels);

    return centers;
}



std::vector<std::vector<double>> kmeans_pp(std::vector<std::vector<double>> &P,
                                           std::vector<double> &point_distances,
                                           std::vector<int> &labels,
                                           std::vector<double> &cumsums,
                                           int k){
    /*
     * kmeans++ initialization
     */

    std::vector<std::vector<double>> centers;

    //choose first center uniformly at random
    double randnr = rand()*P.size()/RAND_MAX;

    int randint = ((int) randnr);

    centers.push_back(P[randint]);

    while(centers.size() < k) {


        update_labels_init(P, centers, labels, point_distances, cumsums);

        int new_center = choose_center(point_distances, cumsums.back());

        centers.push_back(P[new_center]);
    }

    update_labels_init(P, centers, labels, point_distances, cumsums);

    return centers;
}

void write_centers_to_file(std::vector<std::vector<double>> v, std::string filename){

    std::ofstream outfile;
    outfile.open(filename);

    for (int i = 0; i < v.size(); ++i) {
        for (int j = 0; j < v[i].size(); ++j) {
            double current_coord = v[i][j];
            outfile << current_coord << " ";
        }
        outfile << std::endl;
    }
}


void write_labels_to_file(std::vector<int> labels, std::string filename){
    std::ofstream outfile;
    outfile.open(filename);

    for (int i = 0; i < labels.size(); ++i) {
        outfile << labels[i] << " ";
        outfile << std::endl;
    }
}



std::vector<int> histogram(double max){
    long long bucketsize = max/100;

    std::vector<int> buckets(100);
    //srand((unsigned) time(NULL));
    double number = rand()*max/RAND_MAX;
    for (int i = 0; i < 10000; ++i) {
        srand((unsigned) time(NULL));
        double randnr = rand()*max/RAND_MAX;
        if(randnr != number){
            std::cout << "Not equal";
        }
        int position = randnr/bucketsize;
        buckets[position]++;

    }
    return buckets;
}



int main(int argc, char** argv){
/*
 Can be called with 3 arguments:    filename (string containing the path to instance file)
                                    n_centers (int specifying number of centers)
                                    algorithm (string specifying which of the algorithms should be run: (1) for k-means
                                                                                                        (2) for greedykm
                                                                                                        (3) for ls++
                                                                                                        (4) for fls++)
 */

    //srand(time(NULL)); // initialize random seed generator

    std::string filename = argv[1];
    int n_centers = std::stoi(argv[2]);
    //third
    std::string algorithm = argv[3];
    
    // maximum number of lloyd steps which are performed in the algorithm
    int max_number_iterations = -1;
    max_number_iterations = std::stoi(argv[4]);
    
    // number of greedy sample steps in D2-sampling. If -1 we use default l=2+log(k)
    int greedy_sample_steps;
    greedy_sample_steps = std::stoi(argv[5]);

    // Number of local search steps which are performed (only additional argument for local search) 
    int local_search_steps;
    local_search_steps = std::stoi(argv[6]);

    std::size_t seed;
    seed = std::stoull(argv[7]);

    // ------------------ Normal Kmeans ----------------------------
    if (algorithm == "1"){
        std::cout << "========================================================================" << std::endl;
        std::cout << filename << std::endl;
        std::cout << "1" << std::endl;
        std::cout << n_centers << std::endl;
        KMEANS my_kmeans(filename, seed, ' ', max_number_iterations);
        output_algorithm kmeans_output = my_kmeans.algorithm(n_centers);
        std::cout << kmeans_output;
    }
    // ------------------ Greedy Kmeans ----------------------------
    if (algorithm == "2"){
        std::cout << "========================================================================" << std::endl;
        std::cout << filename << std::endl;
        std::cout << "2" << std::endl;
        std::cout << n_centers << std::endl;
        GREEDY_KMEANS my_greedy_kmeans(filename, seed, ' ', max_number_iterations, greedy_sample_steps);
        output_algorithm greedy_output = my_greedy_kmeans.algorithm(n_centers);
        std::cout << greedy_output;
    }
    // ----------------- Local Search -------------------------------
    if (algorithm == "3"){
        std::cout << "========================================================================" << std::endl;
        std::cout << filename << std::endl;
        std::cout << "3 " << local_search_steps << std::endl;
        std::cout << n_centers << std::endl;
        LOCAL_SEARCH my_local_search(filename, seed, ' ', max_number_iterations, greedy_sample_steps, local_search_steps);
        output_algorithm local_search_output = my_local_search.algorithm(n_centers);
        std::cout << local_search_output;
    }
    // ----------------- FLSPP ---------------------------------------
    if (algorithm == "4"){
        std::cout << "========================================================================" << std::endl;
        std::cout << filename << std::endl;
        std::cout << "4 " << local_search_steps << std::endl;
        std::cout << n_centers << std::endl;
        FLSPP my_flspp(filename, seed, ' ', max_number_iterations, greedy_sample_steps, local_search_steps);
        output_algorithm flspp_output = my_flspp.algorithm(n_centers);
        std::cout << flspp_output;
    }

    // ----------------- Initialization only ---------------------------------------
    if (algorithm == "5"){
        std::cout << filename << std::endl;
        GREEDY_KMEANS my_greedy_kmeans(filename, ' ', 0, greedy_sample_steps);
        output_algorithm init_output = my_greedy_kmeans.algorithm(n_centers);
        for (int i = 0; i < init_output.final_centers.size(); ++i) {
            for (int j = 0; j < init_output.final_centers[i].dimension; ++j) {
                std::cout << init_output.final_centers[i].coordinates[j] << " ";
            }
            std::cout << std::endl;
        }
    }


    /*
    if (algorithm == "1"){
        std::cout << "========================================================================" << std::endl;
        std::cout << "========================================================================" << std::endl;
        std::cout << "Dataset: " << filename << std::endl;
        std::cout << "Algorithm: k-means" << std::endl;
        std::cout << "k = " << n_centers << std::endl;
        KMEANS my_kmeans(filename);
        output_algorithm kmeans_output = my_kmeans.algorithm(n_centers);
        std::cout << kmeans_output << std::endl;
    }

    if (algorithm == "2"){
        std::cout << "========================================================================" << std::endl;
        std::cout << "========================================================================" << std::endl;
        std::cout << "Dataset: " << filename << std::endl;
        std::cout << "Algorithm: Greedy k-means" << std::endl;
        std::cout << "k = " << n_centers << std::endl;
        GREEDY_KMEANS my_greedy_kmeans(filename);
        output_algorithm greedy_output = my_greedy_kmeans.algorithm(n_centers);
        std::cout << greedy_output << std::endl;
    }

    if (algorithm == "3"){
        std::cout << "========================================================================" << std::endl;
        std::cout << "========================================================================" << std::endl;
        std::cout << "Dataset: " << filename << std::endl;
        std::cout << "Algorithm: LS++" << std::endl;
        std::cout << "k = " << n_centers << std::endl;
        LOCAL_SEARCH my_local_search(filename);
        output_algorithm local_search_output = my_local_search.algorithm(n_centers);
        std::cout << local_search_output << std::endl;
    }

    if (algorithm == "4"){
        std::cout << "========================================================================" << std::endl;
        std::cout << "========================================================================" << std::endl;
        std::cout << "Dataset: " << filename << std::endl;
        std::cout << "Algorithm: ForesightLS++" << std::endl;
        std::cout << "k = " << n_centers << std::endl;
        FLSPP my_flspp(filename);
        output_algorithm flspp_output = my_flspp.algorithm(n_centers);
        std::cout << flspp_output << std::endl;
    }
    */
    return 0;
}
