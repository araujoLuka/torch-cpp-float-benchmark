#!/bin/bash

# This script is used to set up the environment for the project.
# It installs the required packages and sets up the environment variables.

# ---------------------------------------------------------------
# Global Variables
# ---------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.6.0%2Bcpu.zip"
LIBTORCH_DIR="$SCRIPT_DIR/../include"

# ---------------------------------------------------------------
# Install Functions
# ---------------------------------------------------------------
function install_cmake() {
    # Installs CMake using apt package manager
    # Returns:
    #   0  - Success
    #   1  - Failed to update package list
    #   2  - Failed to install CMake

    if sudo apt update; [ $? -ne 0 ]; then
        return 1
    fi

    if sudo apt install -y cmake; [ $? -ne 0 ]; then
        return 2
    fi

    return 0
}

function install_libtorch() {
    # Installs libtorch from the official PyTorch repository
    # Returns:
    #   0  - Success
    #   1  - Failed to create directory
    #   2  - Failed to download libtorch
    #   3  - Failed to unzip libtorch

    # Create the directory if it doesn't exist
    mkdir -p "$LIBTORCH_DIR"
    if [ $? -ne 0 ]; then
        return 1
    fi

    # Download and unzip libtorch
    wget "$LIBTORCH_URL" -O /tmp/libtorch.zip
    if [ $? -ne 0 ]; then
        return 2
    fi

    unzip /tmp/libtorch.zip -d "$LIBTORCH_DIR"
    if [ $? -ne 0 ]; then
        return 3
    fi

    rm /tmp/libtorch.zip
    return 0
}

function install_likwid() {
    # Installs Likwid using apt package manager
    # Returns:
    #   0  - Success
    #   1  - Failed to update package list
    #   2  - Failed to install Likwid

    if sudo apt update; [ $? -ne 0 ]; then
        return 1
    fi

    if sudo apt install -y likwid; [ $? -ne 0 ]; then
        return 2
    fi

    return 0
}

# ---------------------------------------------------------------
# Check Functions
# ---------------------------------------------------------------
function check_for_libtorch() {
    # Checks if libtorch is installed
    # Returns:
    #   0  - Libtorch is installed
    #   1  - Libtorch is not installed

    if [ ! -d "$LIBTORCH_DIR/libtorch" ]; then
        return 1
    fi
    return 0
}

function check_for_cmake() {
    # Checks if CMake is installed
    # Returns:
    #   0  - CMake is installed
    #   1  - CMake is not installed

    if command -v cmake &> /dev/null; [ $? -ne 0 ]; then
        return 1
    fi
    return 0
}

function check_for_likwid() {
    # Checks if Likwid is installed
    # Returns:
    #   0  - Likwid is installed
    #   1  - Likwid is not installed

    if command -v likwid &> /dev/null; [ $? -ne 0 ]; then
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------
# Main Script
# ---------------------------------------------------------------
function main {
    # Main function to check and install required packages
    # Returns:
    #   0  - Success
    #   1  - Script not run as root
    #   2  - Failed to install CMake
    #   3  - Failed to install Libtorch
    local error_code=0

    echo "Script Directory: $SCRIPT_DIR"
    echo "Libtorch URL: $LIBTORCH_URL"
    echo "Libtorch Directory: $LIBTORCH_DIR"

    echo "----"
    echo "Pre-setup environment for *torch-cpp-float-benchmark* project"
    echo "----"

    echo "Starting pre-setup script..."

    # Check if the script is run as root
    if [ "$EUID" -ne 0 ]; then
        echo "WARNING: Please run this script as root or with sudo."
        echo "Executing with sudo..."
        sudo "$0" "$@"
        exit $?
    fi

    if check_for_cmake; then
        echo "CMake is already installed."
    else
        echo "CMake not found. Installing..."
        if install_cmake; then
            echo "CMake installation completed successfully."
        else
            error_code=$?
            echo "ERROR: CMake installation failed with error code $error_code."
            case $error_code in
                1)
                    echo -e "\t- Failed to update package list."
                    ;;
                2)
                    echo -e "\t- Failed to install CMake."
                    ;;
                *)
                    echo -e "\t- Unknown error occurred."
                    ;;
            esac
            exit 2
        fi
    fi

    if check_for_libtorch; then
        echo "Libtorch is already installed."
    else
        echo "Libtorch not found. Installing..."
        if install_libtorch; then
            echo "Libtorch installed successfully."
        else
            error_code=$?
            echo "ERROR: Libtorch installation failed with error code $error_code."
            case $error_code in
                1)
                    echo -e "\t- Failed to create directory."
                    ;;
                2)
                    echo -e "\t- Failed to download libtorch."
                    ;;
                3)
                    echo -e "\t- Failed to unzip libtorch."
                    ;;
                *)
                    echo -e "\t- Unknown error occurred."
                    ;;
            esac
            exit 3
        fi
    fi

    if check_for_likwid; then
        echo "Likwid is already installed."
    else
        echo "Likwid not found. Installing..."
        if install_likwid; then
            echo "Likwid installation completed successfully."
        else
            error_code=$?
            echo "ERROR: Likwid installation failed with error code $error_code."
            case $error_code in
                1)
                    echo -e "\t- Failed to update package list."
                    ;;
                2)
                    echo -e "\t- Failed to install Likwid."
                    ;;
                *)
                    echo -e "\t- Unknown error occurred."
                    ;;
            esac
            exit 4
        fi
    fi

    echo "All required packages installed successfully."
}

# Run the main function
main
