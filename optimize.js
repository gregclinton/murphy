/*global angular*/

angular.module('main', []);

angular.module('main').controller('optimize', ['$scope', '$http', function ($scope, $http) {
    $scope.charts = [];

    $http({ url: '/optimize/charts' }).then(
        function (res) {
            var charts = res.data,
                id = 1000000;   

            charts.forEach(function (chart) { chart.id = id; id += 1; });
            $scope.charts = charts;
        },

        function (trace) {        
            $scope.error = trace;
        }
    );
}]);