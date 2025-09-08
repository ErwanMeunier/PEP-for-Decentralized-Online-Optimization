% Written by ChatGPT 5 
function minVal = smallestNonNull(A, threshold)
    % smallestNonNull returns the smallest entry of a matrix
    % that is greater than a given threshold (ignores near-zero values).
    %
    % Usage:
    %   minVal = smallestNonNull(A, threshold)
    %
    % Input:
    %   A         - input matrix
    %   threshold - minimum absolute value to consider
    %
    % Output:
    %   minVal - the smallest value in A whose absolute value > threshold

    % Keep only elements above threshold (ignoring near-zero entries)
    filteredElements = A(abs(A) > threshold);

    % If none found, return empty
    if isempty(filteredElements)
        minVal = [];
        warning('No elements found above the given threshold.');
        return;
    end

    % Get smallest among filtered values
    minVal = min(filteredElements);
end