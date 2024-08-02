cd "$(dirname "$(pwd)")"
while [ ! -d ".git" ]; do
    if [ "$(pwd)" = "/" ]; then
        echo "Kein Git-Repository gefunden"
        exit 1
    fi
    cd "$(dirname "$(pwd)")"
done
echo "Git-Repository gefunden in: $(pwd)"
