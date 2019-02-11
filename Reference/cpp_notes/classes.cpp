#include <iostream>
#include <string>
/*
struct Robot_Struct
{
    int id;
    int no_wheels;
    std::string robot_name;
};

int main()
{
    Robot_Struct robot_1;
    robot_1.id = 2;
    robot_1.robot_name = "Poopy Face";
    std::cout << robot_1.robot_name << std::endl;
    return 0;
}
*/



class Robot_Class
{

public:
    int id;
    int no_wheels;

    std::string robot_name;

    // Note: it is standard practice to only declare funcs in class
    //   and define outside of class, to keep class def short
    void move_robot();
    void stop_robot();
};

// Now we define the class funcs

void Robot_Class::move_robot()
{
    std::cout << "Moving Robot!" << std::endl;
}

void Robot_Class::stop_robot()
{
    std::cout << "Stopping Robot!" << std::endl;
}


/*
int main()
{
    Robot_Class robot;
    robot.id = 4;
    robot.robot_name = "Winston";

    robot.move_robot();
    robot.stop_robot();

    return 0;
}
*/


// Listing 2-3
//  inheritance


class Robot_Class_Derived: public Robot_Class
{
public:
    // note: id, robot_name, move_robot, stop_robot not defined!
    void turn_left();
    void turn_right();
};

void Robot_Class_Derived::turn_left()
{
    std::cout << "Robot Turn Left" << std::endl;
}

void Robot_Class_Derived::turn_right()
{
    std::cout << "Robot Turn Right" << std::endl;
}


int main()
{
    Robot_Class_Derived robot;

    robot.id = 2;
    robot.robot_name = "Mobile robot";

    std::cout << "Robot ID=" << robot.id << std::endl;
    std::cout << "Robot Name=" << robot.robot_name << std::endl;

    robot.move_robot();
    robot.stop_robot();

    robot.turn_left();
    robot.turn_right();

    return 0;
}
