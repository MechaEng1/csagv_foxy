import sys
import rclpy
import numpy as np
from casadi import *
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
from casadi import SX
from do_mpc.model import Model
import do_mpc

class NMPCNode(Node):
    def __init__(self,):
        super().__init__('nmpc_node')

        # Intervallo di tempo tra un passo e l'altro
        self.dt = 0.1  # Secondi
        # Numero di passi nell'orizzonte di predizione
        self.N = 20

        # Stato iniziale del robot (x, y, theta)
        self.x0 = np.zeros((3, 1))  # [x, y, theta]
        # Controlli iniziali (velocità lineare e angolare)
        self.u0 = np.zeros((2, self.N))  # [v, omega]
        # Posizione di riferimento (target)
        self.x_ref = np.zeros((3, 1))  # [x_ref, y_ref, theta_ref]

        # Definizione del modello dinamico del robot
        self.model = self.define_model()

        # Configurazione dell'ottimizzatore NMPC
        self.mpc = self.define_mpc()

        # Publisher per inviare comandi al robot
        self.control_publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        # Subscriber per ricevere la posizione odometrica del robot
        self.odom_subscription = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        # Subscriber per ricevere la posizione del cono
        self.cone_subscription = self.create_subscription(Point, "/cone_position", self.cone_callback, 10)

        # Timer per eseguire il calcolo NMPC periodicamente
        self.timer = self.create_timer(self.dt, self.nmpc_callback)

    def define_model(self):
        # Creazione del modello continuo a uniciclo

        model_type = 'continuous'  # Tipo di modello
        model = Model(model_type)


        # Variabili di stato
        L = 0.5
        x = model.set_variable(var_type='_x', var_name='x')  # Posizione x
        y = model.set_variable(var_type='_x', var_name='y')  # Posizione y
        theta = model.set_variable(var_type='_x', var_name='theta')  # Orientamento

        # Variabili di controllo
        v = model.set_variable(var_type='_u', var_name='v')  # Velocità lineare
        omega = model.set_variable(var_type='_u', var_name='omega')  # Velocità angolare

        # Equazioni dinamiche del modello uniciclo
        model.set_rhs('x', v * cos(theta))  # Derivata di x
        model.set_rhs('y', v * sin(theta))  # Derivata di y
        model.set_rhs('theta', v/L*tan(omega))  # Derivata di theta

        # Configurazione del modello
        model.setup()
        return model

    def define_mpc(self):
        # Creazione dell'oggetto MPC basato sul modello
        mpc = do_mpc.controller.MPC(self.model)

        # Configurazione dei parametri NMPC
        setup_mpc = {
            'n_horizon': self.N,  # Numero di passi nell'orizzonte
            't_step': self.dt,  # Intervallo di tempo tra i passi
            'state_discretization': 'collocation',  # Metodo di discretizzazione
            'collocation_type': 'radau',  # Tipo di collocazione
            'store_full_solution': True,  # Memorizzazione della soluzione completa
            'nlpsol_opts': {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}  # Opzioni per il solver
        }
        mpc.set_param(**setup_mpc)

        # Funzione di costo per obstacle avoidance
        safe_distance = 1.0  # Distanza di sicurezza dal cono
        penalty_factor = 50  # Penalità per violazione della distanza

        # Costo terminale: Minimizza la distanza dal punto obiettivo (passando intorno al cono)
        mpc.set_objective(
            mterm=((self.model.x['x'] - self.x_ref[0])**2 + (self.model.x['y'] - self.x_ref[1])**2),
            lterm=(
                # Penalità sulla distanza dal cono
                ((safe_distance / (sqrt((self.model.x['x'] - self.x_ref[0])**2 + (self.model.x['y'] - self.x_ref[1])**2) + 1e-5))**2) * penalty_factor +
                # Penalità sui comandi per cambiamenti bruschi
                self.model.u['v']**2 + self.model.u['omega']**2
            )
        )

        # Vincoli sui controlli
        mpc.bounds['lower', '_u', 'v'] = 0.0  # Velocità lineare minima
        mpc.bounds['upper', '_u', 'v'] = 1.5  # Velocità lineare massima
        mpc.bounds['lower', '_u', 'omega'] = -1.0  # Velocità angolare minima
        mpc.bounds['upper', '_u', 'omega'] = 1.0  # Velocità angolare massima

        # Configurazione finale del controller
        mpc.setup()
        mpc.x0 = self.x0  # Stato iniziale del robot
        mpc.set_initial_guess()
        return mpc

    def odom_callback(self, msg):
        # Aggiorna lo stato corrente del robot basato sull'odometria
        self.x0[0] = msg.pose.pose.position.x  # Posizione x
        self.x0[1] = msg.pose.pose.position.y  # Posizione y
        self.x0[2] = msg.pose.pose.orientation.z  # Approssimazione dell'orientamento

    def cone_callback(self, msg):
        # Aggiorna la posizione del cono come riferimento per il controllo
        self.x_ref[0] = msg.x  # Posizione x del cono
        self.x_ref[1] = msg.y  # Posizione y del cono

    def nmpc_callback(self):
        try:
            # Calcola il controllo ottimale usando NMPC
            u0 = self.mpc.make_step(self.x0)

            # Prepara il messaggio Twist per comandare il robot
            msg_cmd = Twist()
            msg_cmd.linear.x = float(u0[0])  # Velocità lineare
            msg_cmd.angular.z = float(u0[1])  # Velocità angolare
            self.control_publisher.publish(msg_cmd)  # Pubblica il comando
        except Exception as e:
            # Logga eventuali errori nel calcolo NMPC
            self.get_logger().error(f"Errore nel calcolo NMPC: {e}")


def main(args=None):
    # Inizializza il nodo ROS2
    rclpy.init(args=args)
    # Crea un'istanza del nodo NMPC
    node = NMPCNode()
    # Mantieni il nodo in esecuzione
    rclpy.spin(node)
    # Arresta il nodo e il sistema ROS2
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()