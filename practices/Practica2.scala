//1. Develop an algorithm in scale that calculates the radius of a circle
printf("Ingrese el diametro del circulo: ")
var diametro =

printf("El radio del circulo es: ")
val radio = (D/2)

//2. Develop an algorithm in scala that tells me if a number is prime

printf("Ingrese un numero: ")
var num =

    if(num%2 == 0){
        println(s"$num es par")
    }else{
        println(s"$num es impar")
    } 

//3. Given the variable var bird = "tweet", use string interpolation to print "I am tweeting"

var im = "Estoy"
var writing = "escribiendo"
var a = "un"
var bird = "tweet"
var sentence = s"${im} ${writing} ${a} ${bird}"
println(sentence)

//4. Given the variable var message = "Hello Luke I am your father!" uses slice to extract the "Luke" sequence

var mensaje = "Hola Luke yo soy tu padre!"
mensaje slice  (5,9)

//5. What is the difference between value (val) and a variable (var) in scala?

//Val es algo inmutable me refiero a que es un valor asignado mientras que var es mutable y se modifica fácilmente.

//6. Given the tuple (2,4,5,1,2,3,3.1416,23) it returns the number 3.1416
val tupla = (2,4,5,1,2,3,3.1416,23)
tupla._7