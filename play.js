/*global angular,Plotly*/

angular.module('main', []);

angular.module('main').controller('main', ['$scope', '$http', function ($scope, $http) {
    $scope.charts = [];
    
    function error(trace) {
        $scope.error = trace;
    }

    $scope.next = function () {
        $http({ url: 'play/next' }).then(
            function (res) {
                Plotly.deleteTraces('chart', 0);
                Plotly.addTraces('chart', { y: res.data, mode: 'lines', line: {color: 'darkred'} });
            }, error);
    };

    $http({ url: 'play/charts' }).then(
        function (res) {
            var traces = [],
                layout = { showlegend: false },
                contour = res.data.contour,
                options = { displayModeBar: false, staticPlot: true };

            min = function (a) { return Math.min.apply(null, a); };
            max = function (a) { return Math.max.apply(null, a); };
            start = min(contour.z.map(min));
            end = max(contour.z.map(max));

            contour.type = 'contour';
            contour.contours = { coloring: 'lines', start: start, end: end, size: (end - start) / 5.0 };
            contour.showscale = false;

            traces.push(contour);
                              
            Plotly.plot('chart', traces, layout, options);
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