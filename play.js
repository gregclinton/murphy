/*global angular,Plotly*/

angular.module('main', []);

angular.module('main').controller('main', ['$scope', '$http', function ($scope, $http) {
    $scope.charts = [];
    
    function error(trace) {
        $scope.error = trace;
    }

    $scope.next = function () {
        $http({ url: 'optimize/next' }).then(
            function (res) {
                Plotly.deleteTraces('chart', 0);
                Plotly.addTraces('chart', { y: res.data, mode: 'lines', line: {color: 'darkred'} });
            }, error);
    };

    $http({ url: 'optimize/charts' }).then(
        function (res) {
            var layout = { xaxis: {}, yaxis: {} },
                trace = {},
                options = { displayModeBar: false, staticPlot: true };

            layout.height = 180;
            layout.width = 320;
            layout.showlegend = false;
            layout.xaxis.showticklabels = false;
            layout.xaxis.ticks = '';
            layout.xaxis.autorange = true;
            layout.xaxis.zeroline = false;
            layout.xaxis.showgrid = false;
            layout.yaxis.autorange = true;
            layout.yaxis.showticklabels = false;
            layout.yaxis.ticks = '';
            layout.yaxis.zeroline = false;
            layout.yaxis.showgrid = false;
            layout.margin = {t: 2, l: 1, r: 1, b: 2};
            
            contour = res.data.contour;
            z = contour.z;
            min = function (a) { return Math.min.apply(null, a); };
            max = function (a) { return Math.max.apply(null, a); };
            start = min(z.map(min));
            end = max(z.map(max));
            
            contour.type = 'contour';
            contour.contours = { coloring: 'lines', start: start, end: end, size: (end - start) / 5.0 };
            contour.showscale = false;

            trace.x = res.data.line.x;
            trace.y = res.data.line.y;
            trace.mode = 'lines';
            trace.line = { color: 'darkblue' };

            Plotly.plot('chart', [contour, trace], layout, options);
        }, error);
}]);

angular.module('main').directive('shortcut', ['$document', function ($document) {
    return {
        restrict: 'E',
        replace: true,
        scope: true,
        link: function (scope, element, attrs) {
            function code(key) {
                return key === 'enter' ? 13 : key.charCodeAt(0);
            }

            $document.bind('keypress', function (e) {
                if (code(attrs.key) === e.which) {
                    scope.$apply(attrs.keypress);
                }
            });
        }
    };
}]);