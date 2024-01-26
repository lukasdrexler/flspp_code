
#include "additional_vector_stuff.h"

// overload comparison of vectors


/*
template<typename T>
bool operator==(const std::vector<T>& lhs, const std::vector<T>& rhs) {
	if (lhs.size() != rhs.size()) return false;
	for (int i = 0; i < lhs.size(); i++) {
		if (lhs[i] != rhs[i]) return false;
	}
	return true;
}
*/


// printing vectors
/*
template <typename T>
std::ostream& operator << (std::ostream& os, const std::vector<T>& vec)
{
	for (auto elem : vec)
	{
		os << elem << " ";
	}
	return os;
}
*/