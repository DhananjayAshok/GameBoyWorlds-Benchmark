#!/usr/bin/env bash

declare -A ARGS

REQUIRED_ARGS=("env_dir")

ALLOWED_FLAGS=("${REQUIRED_ARGS[@]}" "${!ARGS[@]}")

USAGE_STR="Usage: $0"

for req in "${REQUIRED_ARGS[@]}"; do
    USAGE_STR+=" --$req <value>"
done

for opt in "${!ARGS[@]}"; do
    if [[ ! " ${REQUIRED_ARGS[*]} " =~ " ${opt} " ]]; then
        if [[ -z "${ARGS[$opt]}" ]]; then
            echo "DEFAULT VALUE OF KEY \"$opt\" CANNOT BE BLANK"
            exit 1
        fi
        USAGE_STR+=" [--$opt <value> (default: ${ARGS[$opt]})]"
    fi
done

function usage() {
    echo "$USAGE_STR"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --*)
            FLAG=${1#--}
            VALID=false
            for allowed in "${ALLOWED_FLAGS[@]}"; do
                if [[ "$FLAG" == "$allowed" ]]; then
                    VALID=true
                    break
                fi
            done
            if [ "$VALID" = false ]; then
                echo "Error: Unknown flag --$FLAG"
                usage
            fi
            ARGS["$FLAG"]="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

for req in "${REQUIRED_ARGS[@]}"; do
    echo $req : "${ARGS[$req]}"
    if [[ -z "${ARGS[$req]}" ]]; then
        echo "Error: Argument --$req is required."
        FAILED=true
    fi
done

if [ "$FAILED" = true ]; then usage; fi

echo "Script: $0 Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done

uv --version || { echo "uv is not installed. Please install uv"; exit 1; }

if [ -d "llm-utils/setup/.venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    echo "Creating virtual environment..."
    if [ -d "${ARGS[env_dir]}" ]; then
        read -p "Directory ${ARGS[env_dir]} already exists. Do you want to delete it and create a new virtual environment? (y/n) " yn
        case $yn in
            [Yy]* ) rm -rf "${ARGS[env_dir]}";;
            * ) echo "Exiting without creating virtual environment."; exit 1;;
        esac
    fi
    uv venv "${ARGS[env_dir]}" --python 3.12 || { echo "Failed to create virtual environment. Please check the error messages above."; exit 1; }
    ln -s "${ARGS[env_dir]}" llm-utils/setup/.venv || { echo "Failed to create symbolic link for virtual environment. Please check the error messages above."; exit 1; }
fi

cd llm-utils/setup
uv sync || { echo "Failed to sync virtual environment. Please check the error messages above."; exit 1; }
echo "Environment setup complete!"