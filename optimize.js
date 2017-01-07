/*global angular*/

angular.module('explore').controller('optimize', ['$scope', function ($scope) {
    $scope.charts = [];

    $scope.get('/optimize/chart', function (charts) {
        var id = 1000000;   
        
        charts.forEach(function (chart) { chart.id = id; id += 1; });
        $scope.charts = charts;
    });
}]);