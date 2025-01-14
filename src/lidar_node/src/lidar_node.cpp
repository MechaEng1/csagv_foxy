// Nodo LiDAR
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/centroid.h>

class LidarNode : public rclcpp::Node {
public:
    LidarNode() : Node("lidar_node") {
        pointcloud_subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/lidar_points", 10, std::bind(&LidarNode::lidar_callback, this, std::placeholders::_1));

        cone_position_publisher_ = this->create_publisher<geometry_msgs::msg::Point>("/cone_position", 10);
    }

private:
    void lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::fromROSMsg(*msg, *cloud);

        // Filtraggio della nuvola di punti (ad esempio, asse Z per rimuovere il terreno)
        pcl::PassThrough<pcl::PointXYZI> pass;
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(0.1, 2.0);
        pass.filter(*cloud);

        // Clustering per identificare ostacoli
        pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>());
        tree->setInputCloud(cloud);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
        ec.setClusterTolerance(0.5);
        ec.setMinClusterSize(10);
        ec.setMaxClusterSize(1000);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(cluster_indices);

        for (const auto &indices : cluster_indices) {
            pcl::PointCloud<pcl::PointXYZI>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZI>());
            for (const auto &idx : indices.indices) {
                cluster->push_back((*cloud)[idx]);
            }

            // Calcola il centroide del cluster
            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(*cluster, centroid);

            // Pubblica la posizione del cono
            geometry_msgs::msg::Point cone_position;
            cone_position.x = centroid[0];
            cone_position.y = centroid[1];
            cone_position.z = centroid[2];
            cone_position_publisher_->publish(cone_position);
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_subscription_;
    rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr cone_position_publisher_;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LidarNode>());
    rclcpp::shutdown();
    return 0;
}