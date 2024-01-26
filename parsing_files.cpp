#include "parsing_files.h"

char get_delimiter(std::string filepath) {
    // collect delimiter from file delimiters in datasets folder
    std::string line;
    std::ifstream infile("datasets/delimiters.txt");


    if (not infile) {
        throw std::runtime_error("Cannot open file.");
    }

    // find dataset name
    
    std::vector<std::string> substrings;

    
    //char delim_for_file = '/';

    // because some delimeters do not really seem to work consistent we test them all...
    std::vector<char> delims{ '/', '\\' };

// not sure why this does not always work... (works on my working pc but not on my home pc, both windows)
//#ifdef OS_Windows
//    delim_for_file = '\\';
//#endif // OS_Windows

    std::string dataset;

    for (int i = 0; i < delims.size(); i++) {
        std::istringstream iss(filepath);
        std::string token;
        std::string current_dataset;
        while (std::getline(iss, token, delims[i])) {
            substrings.push_back(token);
        }
        current_dataset = substrings.back();
        if (i == 0 || current_dataset.size() < dataset.size()) dataset = current_dataset;

        substrings.clear();
    }

    

    // iterate through list until delimiter is found

    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        substrings.clear();

        while (!ss.eof()) {
            std::string current_string;
            getline(ss, current_string, '-');            
            substrings.push_back(current_string);
        }

        if (substrings[0] == dataset) { // we found correct dataset name
            return substrings[1][0];
        }
    }

    std::cout << "Could not find dataset delimiter for dataset " << dataset << " !!" << std::endl;
    std::cout << "Using \" \" " << std::endl;
    return ' ';
}

std::vector<Point> read_file(std::string filepath, char delimiter) {
    std::string line;
    std::ifstream infile(filepath);


    if (not infile) {
        throw std::runtime_error("Cannot open file.");
    }

    std::vector<Point> points;
    int counter = 0;
    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        std::vector<double> coordinates;

        while (!ss.eof()){
            std::string coordinate_string;
            getline(ss, coordinate_string, delimiter);
            double coordinate = std::stod(coordinate_string);
            coordinates.push_back(coordinate);
        }

        int dim = coordinates.size();
        Point p = Point(dim, counter, coordinates);
        points.push_back(p);
        counter++;
    }
    //std::vector<double> x = points[44585].coordinates;
    return points;
}