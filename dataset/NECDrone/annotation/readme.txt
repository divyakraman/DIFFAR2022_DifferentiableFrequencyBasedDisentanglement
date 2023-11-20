1. Annotation file data format
Every annotation file (csv format) consists of 7 columns. The following colums are as follows.

1) The drone ID: either NEC-Phantom-001 or NEC-Phantom-002
2) The action ID: A is single-person action, B is two-person action, following 4 digit is action ID
3) The actor ID: PXXXX, unique for each actor
4) The video ID: VXXXX, unique for each actor
5) Start frame index
6) End frame index
7) Action label (0-based)

The NEC-Drone dataset video frames are organized following the hierarchy in the order of the columns 1)->2)->3>->4).


2. Annotation file prefix
The prefix NEC-Drone-16 stands for the full dataset, and the prefix NEC-Drone-7 stands for the subset where the classes are overlaps with Kinetics-400 dataset. 


3. Class correspondence
Class correspodences between NEC-Drone-7 and Kinetics-400 are as follows:
----------------------------------------------------------------------------------------------------------
NEC-Drone class 		| 	Kinetics class
----------------------------------------------------------------------------------------------------------
walking				| marching
running				| jogging, running on treadmill
jumping				| high jump, jumping into pool
drinking water from a bottle	| drinking beer
throwing an object		| throwing axe, throwing ball, throwing discuss, shot put, javeling throw
shaking hands			| shaking hands
hugging				| hugging
----------------------------------------------------------------------------------------------------------
