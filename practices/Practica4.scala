//1.Algoritmo 1 Versión recursiva descendente
def fibn1(num: Int): Int = {
    if(num<2){
        return num
    }
    else{
        return fibn1(num-1)+fibn1(num-2)
    }
}

//Algoritmo 2 Versión con fórmula explícita
def fibn2(num:Int) : Double={
    if(num<2){return num}
    else{var p=((1+math.sqrt(5))/2)
    var j= ((math.pow(p,num)-(math.pow((1-p),num)))/(math.sqrt(5)))
    return j
    }
}

//Algoritmo 3 Versión iterativa
def fibn3(num:Int) : Int ={
    var a=0
    var b=1
    var c= 0
    var k=0
    while (k < num){
        c=b+a
    a=b
    b=c
    k=k+1
    }
    return a
}
//Algoritmo 4 Versión iterativa 2 variables 
def fibn4(num:Int) : Int ={
    var a=0
    var b=1
    var k= 0

    while (k < num){
        b=b+a
    a=b-a

    k=k+1
    }
    return a
}
//Algoritmo 5 Versión iterativa vector
def fibn5(num:Int) : Int ={
    {
        if(num <2)
        {
            return num;
        }
        else
        {
            var arreglo = Array.ofDim[Int](num+1);
            arreglo(0)=0;
            arreglo(1)=1;
            for(k <- Range(2, num+1))
            {
                arreglo(k)=ar(k-1)+arreglo(k-2);
            }
            return arreglo(num);
        }
    }