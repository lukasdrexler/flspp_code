#pragma once

#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>

#include "clustering.h"
#include "makros.h"


char get_delimiter(std::string filepath);
std::vector<Point> read_file(std::string filepath, char delimiter=' ');
