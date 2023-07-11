#include <memory>
#include <map>
#include <iostream>
#include <vector>

class A {
    public:
        A(int i0):i(i0) { std::cout << "A cstor " << i << std::endl; };
        ~A() { std::cout << "A dstor " << i << std::endl; };
        int i;
};

int main() {

    // map iterator re-assignment
    std::map<int, std::shared_ptr<A> > m;

    m[1] = std::make_shared<A>(1);

    auto iter = m.find(1);
    iter->second = std::make_shared<A>(2);

    std::cout << "assigned" << std::endl;

    // auto reassignment
    auto a = std::make_shared<A>(3);
    a= std::make_shared<A>(4);

    std::cout << "assigned 2" << std::endl;

    // const reassignment
    
    std::shared_ptr<A> b = std::make_shared<A>(5);

    std::shared_ptr<const A> c(b);

    std::cout << "assigned 3" << std::endl;

    std::vector<std::shared_ptr<const A> > vec;
    vec.push_back(b);

    // notice the destruction sequence is in reverse order
    // as local variables in the stack

    // this is compiler error!
    //c->i = 2;
    //vec[0]->i = 3;

    return 0;
}
