% create example data
x = [1 2 3 1 2 3 1 2 3];
y = [1 1 1 2 2 2 3 3 3];
z = [10 20 30 40 50 60 70 80 90];

% set up regular grid
xq = linspace(min(x), max(x), 100);
yq = linspace(min(y), max(y), 100);
[X,Y] = meshgrid(xq,yq);

% interpolate data onto grid
Z = griddata(x, y, z, X, Y);

% create surface plot
surf(X,Y,Z);

