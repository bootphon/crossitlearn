while read w; do
    echo $w
    #say -v $voice -f $file -o ${voice}_${file%.*}.aif
    say $w -o $w.aif
done < words.txt
