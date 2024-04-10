pdb=$1
count=$2
res=$3

fres=$((res - 7))
lres=$((res + 7)) 


sed -ni '/ATOM/p' $pdb
number=$RANDOM

for i in $(seq $fres $lres); 
do
   awk -v var=$i '{if ($5 == var) print $0}' $pdb >> $number.pdb
done

~/programs/pales/linux/pales -inD RDC_virtual_132_NS.txt -pdb $number.pdb -outD $pdb\_$count\_out_pal_126 -lcS 1.0  > $pdb\_$count\_out_pal_126


grep "^[[:blank:]]\+$res[[:blank:]]\+ALA" $pdb\_$count\_out_pal_126 >> $res\_values

rm $pdb\_$count\_out_pal_126 $number.pdb

