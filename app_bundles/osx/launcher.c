#include <stdlib.h>
#include <unistd.h>
#include <libgen.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <mach-o/dyld.h>

int main(int argc, char *argv[]) {
    char exePath[PATH_MAX];
    uint32_t size = sizeof(exePath);
    if (_NSGetExecutablePath(exePath, &size) != 0) {
        fprintf(stderr, "Failed to get executable path\n");
        return 1;
    }

    char *dir = dirname(exePath);
    char command[2048];

    // Escape quotes inside AppleScript to handle paths with spaces
    snprintf(command, sizeof(command),
        "osascript -e 'tell application \"Terminal\" to do script \"cd \\\"%s/../Resources\\\" && ./inference-app\"'",
        dir);

    return system(command);
}
//compile with: clang -target arm64-apple-macos -o launcher launcher.c