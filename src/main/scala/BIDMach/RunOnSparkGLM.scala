package BIDMach

import BIDMach.datasources.IteratorSource
import BIDMach.datasources.IteratorSource.{Options => IteratorOpts}
import BIDMach.models.Model
import BIDMach.updaters.Batch
import BIDMat.MatIO
import BIDMat.SciFunctions._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.hadoop.io.Text;
import org.apache.spark.SparkContext
import scala.reflect.ClassTag
import org.apache.spark.util.SizeEstimator;
import BIDMach.models.KMeans
import BIDMat.{CMat,CSMat,DMat,Dict,FMat,FND,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,GND,HMat,IDict,Image,IMat,LMat,Mat,SMat,SBMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._
import BIDMach.models.{Click,FM,GLM,KMeans,KMeansw,LDA,LDAgibbs,Model,NMF,SFA,RandomForest,SVD}
import BIDMach.networks.{Net}
import BIDMach.datasources.{DataSource,MatSource,FileSource,SFileSource}
import BIDMach.datasinks.{DataSink,MatSink}
import BIDMach.mixins.{CosineSim,Perplexity,Top,L1Regularizer,L2Regularizer}
import BIDMach.updaters.{ADAGrad,Batch,BatchNorm,Grad,IncMult,IncNorm,Telescoping}
import BIDMach.causal.{IPTW}
import java.net.{InetAddress,InetSocketAddress}
import BIDMach.allreduce.{Master,Worker,Command}

object RunOnSparkGLM {

  // def runOnSparkGLM(sc: SparkContext, learner:Learner, rddData:RDD[(Text,MatIO)], num_partitions: Int):Array[Learner] = {
  def runOnSparkGLM(sc: SparkContext, protoLearner:Learner, rddData:RDD[(Text,MatIO)], num_partitions: Int) = {
    // Instantiate a learner, run the first pass, and reduce all of the learners' models into one learner.
    Mat.checkMKL(true)
    Mat.hasCUDA = 0
    Mat.checkCUDA(true)

    val workerCmdPorts = (0 until num_partitions).map(_ => scala.util.Random.nextInt(55532) + 10000 - 2)
    val LOCAL_IP = java.net.InetAddress.getLocalHost.getHostAddress

    val m = new Master();
    val opts = m.opts;
    opts.trace = 3;
    opts.intervalMsec = 2000;
    //opts.limitFctn = Master.powerLimitFctn
    opts.limit = 1000000
    opts.timeScaleMsec = 2e-3f

    m.init

    val rddLearnerWorker:RDD[(Learner, Worker)] = rddData.mapPartitionsWithIndex(
      (idx:Int, data_iter:Iterator[(Text, BIDMat.MatIO)]) => {
        import BIDMat.{CMat,CSMat,DMat,Dict,FMat,FND,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,GND,HMat,IDict,Image,IMat,LMat,Mat,SMat,SBMat,SDMat}
        import BIDMat.MatFunctions._
        import BIDMat.SciFunctions._
        import BIDMat.Solvers._
        import BIDMat.Plotting._
        import BIDMach.Learner
        import BIDMach.models.{Click,FM,GLM,KMeans,KMeansw,LDA,LDAgibbs,Model,NMF,SFA,RandomForest,SVD}
        import BIDMach.networks.{Net}
        import BIDMach.datasources.{DataSource,MatSource,FileSource,SFileSource}
        import BIDMach.datasinks.{DataSink,MatSink}
        import BIDMach.mixins.{CosineSim,Perplexity,Top,L1Regularizer,L2Regularizer}
        import BIDMach.updaters.{ADAGrad,Batch,BatchNorm,Grad,IncMult,IncNorm,Telescoping}
        import BIDMach.causal.{IPTW}
        Mat.checkMKL(true)
        Mat.hasCUDA = 0
        Mat.checkCUDA(true)

        val l = protoLearner
        val i_opts = new IteratorSource.Options
        val iteratorSource = new IteratorSource(i_opts)
        val learner = new Learner(iteratorSource, l.model, l.mixins, l.updater, l.datasink)

        val worker = new Worker();
        val opts = worker.opts;
        val cmdPort = workerCmdPorts(idx);
        opts.trace = 4;
        opts.machineTrace = 1;
        opts.commandSocketNum = cmdPort + 0;
        opts.responseSocketNum = cmdPort + 1;
        opts.peerSocketNum = cmdPort + 2;

        println("-------------------------------")
        println("worker init: " + worker)
        println("-------------------------------")

        Iterator[(Learner, Worker)]((learner, worker))
      },
      preservesPartitioning=true
    ).persist()

    val workerAddrs:Array[InetSocketAddress] = rddLearnerWorker.mapPartitions((iter:Iterator[(Learner, Worker)]) => {
      val (learner, worker) = iter.next

      worker.start(learner)

      println("-------------------------------")
      println("worker start: " + worker)
      println("-------------------------------")

      Iterator[InetSocketAddress](
        new InetSocketAddress(worker.workerIP.getHostAddress, worker.opts.commandSocketNum))
    }, preservesPartitioning=true).collect()

    val nmachines = workerAddrs.length;
    val gmods = irow(nmachines);
    val gmachines = irow(0->nmachines);
    m.config(gmods, gmachines, workerAddrs)
    m.setMachineNumbers
    m.sendConfig
    m.setMachineNumbers // BUG: need to send this twice for it to work

    rddData.zipPartitions(rddLearnerWorker, preservesPartitioning=true)(
      (data_iter:Iterator[(Text, BIDMat.MatIO)], lw_iter:Iterator[(Learner, Worker)]) => {
        val (_, worker) = lw_iter.next
        val learner = worker.learner

        println("-------------------------------")
        println("worker first: " + worker)
        println("-------------------------------")

        learner.datasource.asInstanceOf[IteratorSource].opts.iter = data_iter
        learner.datasource.init
        learner.model.bind(learner.datasource)

        learner.model.refresh = true
        learner.init

        learner.firstPass(null)

        learner.datasource.close
        learner.model.mats = null
        learner.model.gmats = null

        Iterator[Int](0)
      }
    ).collect()

    m.allreduce(0, opts.limit)

    // While we still have more passes to complete
    for (iter <- 1 until protoLearner.opts.npasses) {
      // Call nextPass on each learner and reduce the learners into one learner
      val t0 = System.nanoTime()

      rddData.zipPartitions(rddLearnerWorker, preservesPartitioning=true)(
        (data_iter:Iterator[(Text, BIDMat.MatIO)], lw_iter:Iterator[(Learner, Worker)]) => {
          val (_, worker) = lw_iter.next
          val learner = worker.learner
          println("-------------------------------")
          println("worker next: " + worker)
          println("-------------------------------")

          learner.datasource.asInstanceOf[IteratorSource].opts.iter = data_iter
          learner.datasource.init
          learner.model.bind(learner.datasource)

          learner.nextPass(null)

          learner.datasource.close
          learner.model.mats = null
          learner.model.gmats = null

          Iterator[Int](0)
        }
      ).collect()

      m.allreduce(iter, opts.limit)

      val t1 = System.nanoTime()
      println("Elapsed time iter " + iter + ": " + (t1 - t0)/math.pow(10, 9)+ "s")
    }

    rddLearnerWorker.mapPartitions((lw_iter:Iterator[(Learner, Worker)]) => {
      val (_, worker) = lw_iter.next
      Iterator[Learner](worker.learner)
    }, preservesPartitioning=true).collect()
  }
}
