PATH=${1}
#pf=bs
# pf=hi
for ITEM in graph_challenge/$PATH/*
do
	case "$ITEM" in
	*.log)
    		  echo "$ITEM"
		/usr/bin/cat ./$ITEM | /usr/bin/grep -o 'Runtime  = [0-9]*\.[0-9]*'  | /usr/bin/awk -F: '{ print $1 }' 
    	;;
	esac
	#if [[ ${ITEM}  == *_bs.log ]]; then
	#	echo $ITEM
	#	cat ./$ITEM | grep -o 'time:[0-9]*\.[0-9]*'  | awk -F: '{ print $2 }'
	#	echo ""
	#fi
done



